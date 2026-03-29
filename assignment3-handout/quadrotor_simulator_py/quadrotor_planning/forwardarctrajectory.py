import numpy as np

from numpy import arctan2 as atan2
from numpy import arcsin as asin
from numpy import cos as cos
from numpy import sin as sin

from . import PolynomialTrajectory
from quadrotor_simulator_py.quadrotor_control.state import State
from quadrotor_simulator_py.utils.pose import Pose
from quadrotor_simulator_py.utils.quaternion import Quaternion
from quadrotor_simulator_py.utils.rotation3 import Rotation3


class ForwardArcTrajectory:

    T = 0.0
    coeffs = []
    initialized = False

    def __init__(self, curr_ref, cmd, T):
        """ Calculates the 8th order forward arc motion primitive coefficients
        
        Args:
            curr_ref is a state object for where to sample the trajectory
            cmd is a 3x1 numpy array consisting of [vel_x, omega, vel_z]
            T is the primitive duration
        
        Output:
            coeffs: 4x9 world frame coefficients
        """

        self.T = T
        coeffs_world_frame = np.zeros((4,9))
        self.coeffs = coeffs_world_frame
        self.initialized = True

    def coeffs_bodyf2worldf(self, coeffs, Twb0):
        '''This is an optional function you can call within the __init__ funtion
        to convert coefficients from body frame to world frame.
        Args:
            coeffs: 4x9 numpy array of coefficients in body frame
            Twb0: Pose() object representing the initial pose of the quadrotor
        Output:
            coeffs_world_frame: 4x9 numpy array of coefficients in world frame
        '''
        coeffs_world_frame = np.zeros((4,9))
        # TO-DO : Apply the world frame rotation and translation
        
        return coeffs_world_frame
    
    def integrate_coefficients(self, cin):
        """ This function is an optional helper function you can call within 
            the __init__ function. It integrates the coefficients passed.
            Coefficients are in the form of c0 + c1*t + c2*t*t + ... + c7*t^7.
        
        Args:
            cin: 4x8 matrix of coefficients for x, y, z, yaw
       
        Output:
            cout: 4x9 matrix of coefficients where the entry in
                  s[0,:] consists of a zero.
        """
        cout = np.zeros((4,9))
        return cout

    def calculate_coefficients_from_contraints(self, s, f):
        """ This function is an optional helper function you can call within 
            the __init__ function. Given a initial constraints and final constraints,
            this function calculates the coefficients for the forward arc motion
            primitives. This function calls the optional get_coefficients function.
            Coefficients are in the form of c0 + c1*t + c2*t*t + ... + c7*t^7.
            Constraints are passed in in the body frame.
        
        Args:
            s: initial constraints State() class instance
            f: final constraints State() class instance
       
        Output:
            coeffs: 4x8 matrix of coefficients in the body frame
        """

        # Return body frame coefficients
        coeffs = np.zeros((4,8))
        return coeffs

    def get_coefficients(self, b):
        """ This function is an optional helper function you can call within 
            the calculate_coefficients_from_contraints function. 
            Given the endpoints constraints in vector form (b), this function
            will calculate the solution to Ax=b we discussed in the Quadrotor Planning
            II lecture for forward arc motion primitives. 
        
        Args:
            b: 8x1 np array of endpoint constraints
       
        Output:
            x: 8x1 nparray of coefficients
        """
        x = np.zeros((8,1))
        return x

    def get_ref(self, t):
        """ Returns reference as State() for a multi-axis trajectory.
            The multi-axis trajectory contains single axis trajectories
            for x, y, z, and yaw. These should be derived by querying
            self.coeffs, which stores the coefficients 4x9 numpy matrix.
        
        Args:
            t: time at which the reference should be generated.
       
        Output:
            s: State() instance populated with references
        """

        s = State()
        return s
