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

        z_B = a_des / norm(a_des)
        x_C = np.array([[cos(yaw_ref)], [sin(yaw_ref)], [0]])
        y_B_raw = np.cross(z_B, x_C, axis=0)
        y_B = y_B_raw / norm(y_B_raw)
        x_B = np.cross(y_B, z_B, axis=0)

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

        jerk = flat_ref.jerk
        snap = flat_ref.snap
        yaw = flat_ref.yaw
        dyaw = flat_ref.dyaw
        d2yaw = flat_ref.d2yaw

        x_C = np.array([[cos(yaw)], [sin(yaw)], [0]])
        y_C = np.array([[-sin(yaw)], [cos(yaw)], [0]])

        T = norm(acc_vec_des)
        Tdot = float(z_B.T @ jerk)

        h_w = (1.0 / T) * (jerk - Tdot * z_B)

        n = np.cross(z_B, x_C, axis=0)
        k = norm(n)
        n_dot = np.cross(h_w, x_C, axis=0) + dyaw * np.cross(z_B, y_C, axis=0)
        k_dot = float(y_B.T @ n_dot)
        dy_B = (n_dot - k_dot * y_B) / k

        dx_B = np.cross(dy_B, z_B, axis=0) + np.cross(y_B, h_w, axis=0)

        dR = np.hstack([dx_B, dy_B, h_w])
        Omega = R_des.T @ dR
        angvel_des = np.array([[Omega[2, 1]], [Omega[0, 2]], [Omega[1, 0]]])

        Tddot = float(h_w.T @ jerk) + float(z_B.T @ snap)
        h_alpha = (1.0 / T) * (snap - Tddot * z_B - 2.0 * Tdot * h_w)

        n_ddot = (np.cross(h_alpha, x_C, axis=0)
                  + 2.0 * dyaw * np.cross(h_w, y_C, axis=0)
                  + d2yaw * np.cross(z_B, y_C, axis=0)
                  - dyaw**2 * np.cross(z_B, x_C, axis=0))

        k_ddot = float(dy_B.T @ n_dot) + float(y_B.T @ n_ddot)
        d2y_B = (n_ddot - k_ddot * y_B - 2.0 * k_dot * dy_B) / k

        d2x_B = (np.cross(d2y_B, z_B, axis=0)
                 + 2.0 * np.cross(dy_B, h_w, axis=0)
                 + np.cross(y_B, h_alpha, axis=0))

        d2R = np.hstack([d2x_B, d2y_B, h_alpha])
        M = R_des.T @ d2R
        angacc_des = np.array([[(M[2, 1] - M[1, 2]) / 2.0],
                               [(M[0, 2] - M[2, 0]) / 2.0],
                               [(M[1, 0] - M[0, 1]) / 2.0]])

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
