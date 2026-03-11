import numpy as np

from numpy import arctan2 as atan2
from numpy import arcsin as asin
from numpy import cos as cos
from numpy import sin as sin

from quadrotor_simulator_py.utils.quaternion import Quaternion


class Rotation3:

    def __init__(self, R=None):
        self.R = None

        if R is None:
            self.R = np.eye(3)
        else:
            self.R = R

    def to_euler_zyx(self):
        """ Convert self.R to Z-Y-X euler angles

        Output:
            zyx: 1x3 numpy array containing euler angles.
                The order of angles should be phi, theta, psi, where
                roll == phi, pitch == theta, yaw == psi
        """
        '''
        The rotation matrix R corresponds to Z-Y-X as (Yaw-Pitch-Roll)
        And the combined rotation is the product of the three rotations:
        R = Rz(psi) * Ry(theta) * Rx(phi)
    
        Here for quick ref taken from slides:
        R[2, 0] = -sin(theta)
        R[2, 1] = cos(theta) * sin(phi)
        R[2, 2] = cos(theta) * cos(phi)
        R[1, 0] = sin(psi) * cos(theta)
        R[0, 0] = cos(psi) * cos(theta)

        Really want to consider a refactor that makes psi, theta, and phi
        as their own class property functions so I can use them elsewhere easily.
        '''

        # 1. Yaw (psi) around Z-axis
        psi = atan2(self.R[1, 0], self.R[0, 0])

        # 2. Pitch (theta) around Y-axis
        # We use sqrt(r21^2 + r22^2) to calculate cos(theta) to handle the range [-pi/2, pi/2]
        theta = atan2(-self.R[2, 0], np.sqrt(self.R[2, 1]**2 + self.R[2, 2]**2))

        # 3. Roll (phi) around X-axis
        phi = atan2(self.R[2, 1], self.R[2, 2])

        return np.array([phi, theta, psi])

    @classmethod
    def from_euler_zyx(cls, zyx):
        """ Convert euler angle rotation representation to 3x3
                rotation matrix. The input is represented as 
                np.array([roll, pitch, yaw]).
        Arg:
            zyx: 1x3 numpy array containing euler angles
                Technically stored as X-Y-Z

        Output:
            Rot: 3x3 rotation matrix (numpy)
        """

        '''
        Trying to make this as clean as possible so I can come back and
        refactor with a property decorator and other class methods later.
        '''

        # Get the angles from the input which robotics likes as X-Y-Z as we stored above.
        phi   = zyx[0]
        theta = zyx[1]
        psi   = zyx[2]

        # Get the sines and cosines set for each one
        c_phi = np.cos(phi)
        s_phi = np.sin(phi)
        c_theta = np.cos(theta)
        s_theta = np.sin(theta)
        c_psi = np.cos(psi)
        s_psi = np.sin(psi)

        # Create the combined Rotation Matrix elements from slides
        ## Row 1
        r11 = c_psi * c_theta
        r12 = c_psi * s_theta * s_phi - s_psi * c_phi
        r13 = c_psi * s_theta * c_phi + s_psi * s_phi

        ## Row 2
        r21 = s_psi * c_theta
        r22 = s_psi * s_theta * s_phi + c_psi * c_phi
        r23 = s_psi * s_theta * c_phi - c_psi * s_phi

        ## Row 3
        r31 = -s_theta
        r32 = c_theta * s_phi
        r33 = c_theta * c_phi

        # Make a nice pretty 3x3 matrix
        R = np.array([
            [r11, r12, r13],
            [r21, r22, r23],
            [r31, r32, r33]
        ])

        Rot = cls()
        Rot.R = R
        return Rot

        '''
        Apparently you can do this to be cleaner instead which
        could save an operation or two in real time.

        - Create the object AND pass the matrix in one shot -
        return cls(R) -- which is the same as Rot = Rotation3(R)
        '''

    def roll(self):
        """ Extracts the phi component from the rotation matrix

        Output:
            phi: scalar value representing phi
        """

        '''
        OOOOOk, this makes sense now. TBH, would have like to have done
        these methods earlier in the assignment, THEN, just called the
        methods to get the to_euler_zyx().

        Did it earlier so just look back at to_euler_zyx()
        '''
        return atan2(self.R[2, 1], self.R[2, 2])

    def pitch(self):
        """ Extracts the theta component from the rotation matrix

        Output:
            theta: scalar value representing theta
        """
        '''
        Same here.
        '''
        
        return atan2(-self.R[2, 0], np.sqrt(self.R[2, 1]**2 + self.R[2, 2]**2))

    def yaw(self):
        """ Extracts the psi component from the rotation matrix

        Output:
            theta: scalar value representing psi
        """
        '''
        Same here.
        '''
        
        return atan2(self.R[1, 0], self.R[0, 0])

    @classmethod
    def from_quat(cls, q: Quaternion):
        """ Calculates the 3x3 rotation matrix from a quaternion
                parameterized as (w,x,y,z).

        Output:
            Rot: 3x3 rotation matrix represented as numpy matrix
        """
        '''
        Added the Quaternion type hint for clarity.

        From OCRL class, unit quaternions are needed to be normalized to 1
        and do the Identity Substitution for optimization to minimze floating point drift.

        Nice little proof for future reference if needed in future assignments:

            1. Start with the code implementation for the diagonal element (R[0,0]):
                r11 = 1 - 2 * (y**2 + z**2)

            2. Unit Quaternion Identity (magnitude is always 1):
                1 = w**2 + x**2 + y**2 + z**2

            3. Substitute '1' in the code formula with the identity:
                r11 = (w**2 + x**2 + y**2 + z**2) - 2*y**2 - 2*z**2

            4. Combine like terms:
                r11 = w**2 + x**2 + (y**2 - 2*y**2) + (z**2 - 2*z**2)
                r11 = w**2 + x**2 - y**2 - z**2

            Just need to remember to normalize the quaternion before using it.
            with the Quaternion.normalize() method.
        '''
        # Extract components needed for Quaternion to Rotation3 conversion
        w = q.w()
        x = q.x()
        y = q.y()
        z = q.z()
        
        # Lecture 3, slide 19: Quaternion to Rotation Matrix Optimized
        ## Row 1
        r11 = 1 - 2 * (y**2 + z**2)
        r12 = 2 * (x * y - z * w)
        r13 = 2 * (x * z + y * w)
        
        ## Row 2
        r21 = 2 * (x * y + z * w)
        r22 = 1 - 2 * (x**2 + z**2)
        r23 = 2 * (y * z - x * w)
        
        ## Row 3
        r31 = 2 * (x * z - y * w)
        r32 = 2 * (y * z + x * w)
        r33 = 1 - 2 * (x**2 + y**2)

        R = np.array([
            [r11, r12, r13],
            [r21, r22, r23],
            [r31, r32, r33]
        ])

        Rot = cls()
        Rot.R = R
        return Rot

    def to_quat(self):
        """ Calculates a quaternion from the class variable
                self.R and returns it

        Output:
            q: An instance of the Quaternion class parameterized
                as [w, x, y, z]
        """

        '''
        Shepperd's Algorithm

        Had to look this one up to avoid using scipy or any other helper
        methods to keep to assignment functions, but absolutely come back
        and refactor this somehow, because it's relatively understandable
        but ugly.

        Algorithm checks to see which w,x,y,z is sufficient to not Divide by Zero.
        
        '''

        tr = self.R[0, 0] + self.R[1, 1] + self.R[2, 2]

        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2  # S=4*qw
            w = 0.25 * S
            x = (self.R[2, 1] - self.R[1, 2]) / S
            y = (self.R[0, 2] - self.R[2, 0]) / S
            z = (self.R[1, 0] - self.R[0, 1]) / S
        elif (self.R[0, 0] > self.R[1, 1]) and (self.R[0, 0] > self.R[2, 2]):
            S = np.sqrt(1.0 + self.R[0, 0] - self.R[1, 1] - self.R[2, 2]) * 2  # S=4*qx
            w = (self.R[2, 1] - self.R[1, 2]) / S
            x = 0.25 * S
            y = (self.R[0, 1] + self.R[1, 0]) / S
            z = (self.R[0, 2] + self.R[2, 0]) / S
        elif self.R[1, 1] > self.R[2, 2]:
            S = np.sqrt(1.0 + self.R[1, 1] - self.R[0, 0] - self.R[2, 2]) * 2  # S=4*qy
            w = (self.R[0, 2] - self.R[2, 0]) / S
            x = (self.R[0, 1] + self.R[1, 0]) / S
            y = 0.25 * S
            z = (self.R[1, 2] + self.R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + self.R[2, 2] - self.R[0, 0] - self.R[1, 1]) * 2  # S=4*qz
            w = (self.R[1, 0] - self.R[0, 1]) / S
            x = (self.R[0, 2] + self.R[2, 0]) / S
            y = (self.R[1, 2] + self.R[2, 1]) / S
            z = 0.25 * S

        return Quaternion([w, x, y, z])
