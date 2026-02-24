#!/usr/bin/env python
import numpy as np
from numpy import sin, cos
from numpy.linalg import norm
import yaml

from quadrotor_simulator_py.quadrotor_control.state import State
from quadrotor_simulator_py.quadrotor_control.trackingerror import TrackingError
from quadrotor_simulator_py.quadrotor_model.mixer import QuadMixer
from quadrotor_simulator_py.quadrotor_control.cascaded_command import CascadedCommand
from quadrotor_simulator_py.utils import Quaternion
from quadrotor_simulator_py.utils import shortest_angular_distance


class QuadrotorPositionControllerPD:

    def __init__(self, yaml_file):
        self.zw = np.array([[0], [0], [1]])  # unit vector [0, 0, 1]^{\top}
        self.gravity_norm = 9.81
        self.current_state = State()
        self.state_ref = State()
        self.tracking_error = TrackingError()

        self.Rdes = np.eye(3)
        self.Rcurr = None
        self.accel_des = 0.0
        self.angvel_des = np.zeros((3, 1))
        self.angacc_des = np.zeros((3, 1))
        self.mass = 0.0

        data = []
        with open(yaml_file, 'r') as stream:
            try:
                data = yaml.safe_load(stream)
            except yaml.YamlError as exc:
                print(exc)

        self.mass = data['mass']

        self._Kx = np.eye(3)
        self._Kx[0, 0] = data['gains']['pos']['x']
        self._Kx[1, 1] = data['gains']['pos']['y']
        self._Kx[2, 2] = data['gains']['pos']['z']

        self._Kv = np.eye(3)
        self._Kv[0, 0] = data['gains']['vel']['x']
        self._Kv[1, 1] = data['gains']['vel']['y']
        self._Kv[2, 2] = data['gains']['vel']['z']

        self.gravity_norm = data['gravity_norm']

    def update_state(self):
        self.Rcurr = self.current_state.rot

    def set_current_state(self, state_in):
        self.current_state = state_in
        self.update_state()

    def get_state(self):
        return self.current_state

    def set_reference_state(self, ref_in):
        self.state_ref = ref_in

    def compute_body_z_accel(self, a_des, R_curr):
        """ Calculates the body-frame z-acceleration

        Args:
            a_des: 3x1 numpy array representing the desired acceleration
            R_curr: 3x3 rotation matrix representing Rwb

        Output:
            u: scalar value representing body-frame z-acceleration
        """

        # Project desired acceleration onto the CURRENT body z-axis.
        # This gives the mass-normalized thrust: c = a_des . (R * e3)
        # It uses R_curr (not R_des) so the thrust accounts for the actual tilt.
        e3 = np.array([[0], [0], [1]])
        u = float(a_des.T @ R_curr @ e3)
        return [u]

    def compute_orientation(self, a_des, yaw_ref):
        """ Calculates the desired orientation

        Args:
            a_des: 3x1 numpy array representing the desired acceleration
            yaw_ref: yaw reference

        Output:
            R_des: 3x3 numpy matrix representing desired orientation
        """

        # ZYX convention (Daoud Eq 4-6): build R_des from thrust direction + yaw.
        #   z_B = a_des / ||a_des||                (thrust direction)
        #   y_c = [-sin(psi), cos(psi), 0]         (heading perpendicular)
        #   x_B = (y_c x z_B) / ||y_c x z_B||     (body x from cross product)
        #   y_B = z_B x x_B                        (completes right-handed frame)
        z_B = a_des / norm(a_des)
        yc = np.array([[-sin(yaw_ref)], [cos(yaw_ref)], [0]])
        x_B_raw = np.cross(yc, z_B, axis=0)
        x_B = x_B_raw / norm(x_B_raw)
        y_B = np.cross(z_B, x_B, axis=0)

        R_des = np.hstack([x_B, y_B, z_B])
        return R_des

    def compute_hod_refs(self, acc_vec_des, flat_ref, R_des):
        """ Calculates the desired angular velocities and accelerations.

        Args:
            acc_vec_des: 3x1 numpy array representing the desired acceleration
            flat_ref: class instance of State() containing the trajectory reference
            R_des: desired rotation

        Output:
            angvel_des: 3x1 numpy array representing desired angular velocity
            angacc_des: 3x1 numpy array representing desired angular acceleration
        """

        x_B = R_des[:, 0:1]
        y_B = R_des[:, 1:2]
        z_B = R_des[:, 2:3]

        j = flat_ref.jerk
        s = flat_ref.snap
        yaw = flat_ref.yaw
        dyaw = flat_ref.dyaw
        d2yaw = flat_ref.d2yaw

        # Heading vectors in world frame (xc points along yaw, yc perpendicular)
        xc = np.array([[cos(yaw)], [sin(yaw)], [0]])
        yc = np.array([[-sin(yaw)], [cos(yaw)], [0]])

        # c = ||a_des|| is the mass-normalized thrust magnitude.
        # c_dot = z_B^T * jerk is its time derivative (Daoud Eq 11).
        c = norm(acc_vec_des)
        c_dot = float(z_B.T @ j)

        # Angular velocities from differential flatness (Daoud Sec VI-B).
        # These come from projecting jerk onto body axes and dividing by thrust:
        #   omega_x = -(y_B . j) / c    (roll rate from lateral jerk)
        #   omega_y =  (x_B . j) / c    (pitch rate from forward jerk)
        omega_x = float(-y_B.T @ j) / c
        omega_y = float(x_B.T @ j) / c

        # omega_z has a CRITICAL coupling term (Daoud Eq 22):
        #   omega_z = (psi_dot * xc.xB + omega_y * yc.zB) / ||yc x zB||
        # The second term (omega_y * yc.zB) is nonzero even when psi_dot=0.
        # It captures how pitch rate changes the heading when the body is tilted.
        k_yc = norm(np.cross(yc, z_B, axis=0))
        xc_xB = float(xc.T @ x_B)
        yc_zB = float(yc.T @ z_B)
        omega_z = (dyaw * xc_xB + omega_y * yc_zB) / k_yc

        angvel_des = np.array([[omega_x], [omega_y], [omega_z]])

        # Angular accelerations from snap (Daoud Sec VI-C).
        # Derived by differentiating the jerk equation s = c_ddot*zB + 2*c_dot*zB_dot + c*zB_ddot
        # and projecting onto body axes:
        #   alpha_y = (xB.s - 2*c_dot*wy - c*wx*wz) / c
        #   alpha_x = (-yB.s - 2*c_dot*wx + c*wy*wz) / c
        # The cross-terms (wx*wz, wy*wz) arise from centripetal coupling in the rotating frame.
        x_B_s = float(x_B.T @ s)
        y_B_s = float(y_B.T @ s)

        alpha_y = (x_B_s - 2.0 * c_dot * omega_y
                   - c * omega_x * omega_z) / c
        alpha_x = (-y_B_s - 2.0 * c_dot * omega_x
                   + c * omega_y * omega_z) / c

        # alpha_z (Daoud final formula, derived from differentiating Eq 22).
        # Includes yaw acceleration, yaw-rate couplings, and tilt-yaw couplings
        # (alpha_y * yc.zB and -wx*wy * yc.yB terms are nonzero even at zero yaw rate).
        xc_yB = float(xc.T @ y_B)
        xc_zB = float(xc.T @ z_B)
        yc_yB = float(yc.T @ y_B)

        alpha_z = (1.0 / k_yc) * (
            d2yaw * xc_xB
            + dyaw * omega_z * xc_yB
            - 2.0 * dyaw * omega_y * xc_zB
            + alpha_y * yc_zB
            - omega_x * omega_y * yc_yB
            - omega_z * (-dyaw * xc_yB + omega_x * yc_zB)
        )

        angacc_des = np.array([[alpha_x], [alpha_y], [alpha_z]])

        return (angvel_des, angacc_des)

    def compute_command(self):
        """ This function contains the following functionality:
                1. Computes the PD feedback-control terms from the position
                   and velocity control errors.
                2. Computes the desired rotation using compute_orientation.
                3. Applies the thrust command to the body frame using
                   compute_body_z_accel
                4. Calculates the desired angular velocities and accelerations.
        """

        # PD position control: desired acceleration = feedforward + gravity comp + PD feedback.
        # a_des = a_ref + g*e3 - Kp*(p - p_ref) - Kd*(v - v_ref)
        # This vector points in the direction the thrust should act (includes gravity).
        e_pos = self.current_state.pos - self.state_ref.pos
        e_vel = self.current_state.vel - self.state_ref.vel

        a_des = (self.state_ref.acc
                 + self.gravity_norm * self.zw
                 - self._Kx @ e_pos
                 - self._Kv @ e_vel)

        self.Rdes = self.compute_orientation(a_des, self.state_ref.yaw)
        self.accel_des = self.compute_body_z_accel(a_des, self.Rcurr)[0]
        self.angvel_des, self.angacc_des = self.compute_hod_refs(
            a_des, self.state_ref, self.Rdes)

    def get_cascaded_command(self):
        casc_cmd = CascadedCommand()
        casc_cmd.thrust_des = self.mass * self.accel_des
        casc_cmd.Rdes = self.Rdes
        casc_cmd.angvel_des = self.angvel_des
        casc_cmd.angacc_des = self.angacc_des
        return casc_cmd

    def get_tracking_error(self):
        return self.tracking_error

    def update_tracking_error(self):
        self.tracking_error = TrackingError()
        self.tracking_error.pos_des = self.state_ref.pos
        self.tracking_error.vel_des = self.state_ref.vel
        self.tracking_error.acc_des = self.state_ref.acc
        self.tracking_error.jerk_des = self.state_ref.jerk
        self.tracking_error.snap_des = self.state_ref.snap
        self.tracking_error.yaw_des = self.state_ref.yaw
        self.tracking_error.dyaw_des = self.state_ref.dyaw
        self.tracking_error.pos_err = self.current_state.pos - self.state_ref.pos
        self.tracking_error.vel_err = self.current_state.vel - self.state_ref.vel
        self.tracking_error.yaw_err = shortest_angular_distance(
            self.current_state.yaw, self.state_ref.yaw)
        self.tracking_error.dyaw_err = self.current_state.dyaw - self.state_ref.dyaw

    def run_ctrl(self):

        # get updated state
        self.update_state()

        # calculate the command
        self.compute_command()

        # update tracking error
        self.update_tracking_error()
