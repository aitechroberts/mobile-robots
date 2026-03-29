import sys
import copy
import os
import unittest
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle

import os
qs_path = os.path.dirname(os.path.abspath(__file__))+'/..'
sys.path.append(qs_path)

from check_states import check_states
from quadrotor_simulator_py.quadrotor_control.state import State
from quadrotor_simulator_py.utils import Rot3
from quadrotor_simulator_py.utils.pose import Pose
from quadrotor_simulator_py.quadrotor_model import QuadrotorModel
from quadrotor_simulator_py.utils.quaternion import Quaternion
from quadrotor_simulator_py.quadrotor_planning.forwardarctrajectory import ForwardArcTrajectory

def test_get_coefficients():
    
    b = np.array([[8.8817842e-16],
                 [0.0000000e+00],
                 [0.0000000e+00],
                 [0.0000000e+00],
                 [2.5000000e-01],
                 [0.0000000e+00],
                 [0.0000000e+00],
                 [0.0000000e+00]])

    correctx = np.array([[ 8.88178420e-16],
                         [-7.10542736e-15],
                         [ 0.00000000e+00],
                         [ 1.55431223e-15],
                         [ 5.46875000e-01],
                         [-6.56250000e-01],
                         [ 2.73437500e-01],
                         [-3.90625000e-02]])

    curr_ref = State()
    cmd = np.zeros((3,1))
    T = 2
    famp = ForwardArcTrajectory(curr_ref, cmd, T)
    x = famp.get_coefficients(b)
    assert(np.max(abs((x - correctx))) < 1e-01)


def test_integrate_coefficients():
    
    coeffs = np.array([[ 1.00000000e+00 ,-6.51170352e-14 ,-8.52651283e-14 ,-3.97256225e-14,
                -1.10892079e-01,  1.10594382e-01, -4.12357162e-02,  5.46620706e-03],
              [ 8.88178420e-16 , 1.66874572e-14 ,-2.52435490e-29 ,-4.18522802e-15,
                -7.39142988e-01,  8.34194229e-01, -3.33857238e-01,  4.63616656e-02],
              [ 0.00000000e+00 ,-5.68434189e-15 , 0.00000000e+00 , 1.24344979e-15,
                4.37500000e-01 ,-5.25000000e-01 , 2.18750000e-01 ,-3.12500000e-02],
              [ 8.88178420e-16 , 8.52651283e-15 , 0.00000000e+00 ,-1.86517468e-15,
                -6.56250000e-01,  7.87500000e-01, -3.28125000e-01,  4.68750000e-02]])
    
    correct_pos_coeffs = np.array([[ 0.00000000e+00,  1.00000000e+00, -3.25585176e-14, -2.84217094e-14,
                            -9.93140563e-15, -2.21784159e-02,  1.84323970e-02, -5.89081660e-03,
                            6.83275883e-04],
                          [ 0.00000000e+00,  8.88178420e-16,  8.34372860e-15, -8.41451632e-30,
                            -1.04630701e-15, -1.47828598e-01,  1.39032372e-01, -4.76938911e-02,
                            5.79520820e-03],
                          [ 0.00000000e+00,  0.00000000e+00, -2.84217094e-15,  0.00000000e+00,
                            3.10862447e-16,  8.75000000e-02, -8.75000000e-02,  3.12500000e-02,
                            -3.90625000e-03],
                          [ 0.00000000e+00,  8.88178420e-16,  4.26325641e-15,  0.00000000e+00,
                            -4.66293670e-16, -1.31250000e-01,  1.31250000e-01, -4.68750000e-02,
                            5.85937500e-03]])

    curr_ref = State()
    cmd = np.zeros((3,1))
    T = 2
    famp = ForwardArcTrajectory(curr_ref, cmd, T)
    pos_coeffs = famp.integrate_coefficients(coeffs)
    assert(np.max(abs((pos_coeffs - correct_pos_coeffs))) < 1e-01)


def test_calculate_coefficients_from_constraints():
    correct_coeffs = [[ 1.00000000e+00 , -6.51170352e-14, -8.52651283e-14, -3.97256225e-14,
                      -1.10892079e-01,  1.10594382e-01, -4.12357162e-02,  5.46620706e-03],
                    [ 8.88178420e-16 , -1.57992788e-14, -2.52435490e-29,  4.18522802e-15,
                      7.39142988e-01 , -8.34194229e-01,  3.33857238e-01, -4.63616656e-02],
                    [ 0.00000000e+00 , -5.68434189e-15,  0.00000000e+00,  1.24344979e-15,
                      4.37500000e-01 , -5.25000000e-01,  2.18750000e-01, -3.12500000e-02],
                    [ 8.88178420e-16 , -8.52651283e-15,  0.00000000e+00,  1.86517468e-15,
                      6.56250000e-01 , -7.87500000e-01,  3.28125000e-01, -4.68750000e-02]]

    s = State()
    s.pos = np.array([0., 0., 0.]).reshape((3,1))
    s.vel = np.array([1.0000000e+00, 8.8817842e-16, 0.0000000e+00]).reshape((3,1))
    s.acc = np.array([-7.10542736e-14,  4.44089210e-16,  0.00000000e+00]).reshape((3,1))
    s.jerk = np.array([-1.70530257e-13, -5.04870979e-29,  0.00000000e+00]).reshape((3,1))
    s.snap = np.array([-2.27373675e-13, -1.00974196e-28,  0.00000000e+00]).reshape((3,1))
    s.rot = np.array([[1., 0., 0.],
                      [0., 1., 0.],
                      [0., 0., 1.]])
    s.angvel = np.array([0., 0., 0.]).reshape((3,1))
    s.angacc = np.array([0., 0., 0.]).reshape((3,1))
    s.yaw = 0.0
    s.dyaw = 8.881784197000684e-16
    s.d2yaw = 0.0
    s.d3yaw =  0.0

    e = State()
    e.pos = np.array([0., 0., 0.])
    e.vel = np.array([0.82533561, 0.56464247, 0.2]).reshape((3,1))
    e.acc = np.array([-0.16939274,  0.24760068,  0.]).reshape((3,1))
    e.jerk = np.array([-0.07428021, -0.05081782,  0.]).reshape((3,1))
    e.snap = np.array([0., 0., 0.]).reshape((3,1))
    e.rot = np.array([[1., 0., 0.],
                      [0., 1., 0.],
                      [0., 0., 1.]])
    e.angvel = np.array([0., 0., 0.]).reshape((3,1))
    e.angacc = np.array([0., 0., 0.]).reshape((3,1))
    e.yaw =   0.0
    e.dyaw =   0.30000000000000115
    e.d2yaw =  0.0
    e.d3yaw =  0.0
    
    curr_ref = State()
    cmd = np.zeros((3,1))
    T = 2
    famp = ForwardArcTrajectory(curr_ref, cmd, T)
    coeffs = famp.calculate_coefficients_from_contraints(s, e)
    assert(np.max(abs((coeffs - correct_coeffs))) < 1e-01)

def test_forwardarcprimitives():
    with open(qs_path+'/data/famp_states.pkl', 'rb') as file:
        correct = pickle.load(file)

    curr_ref = correct['curr_ref']
    states = []
    for om in np.arange(-1, 1.05, 0.05):
        velx = 1.0
        omega = om
        velz = 0.2

        cmd = np.array([velx, omega, velz]).reshape(3,1)
        T = 2

        famp = ForwardArcTrajectory(curr_ref, cmd, T)
        xs = []
        ys = []
        ref = None
        for t in np.arange(0, T, 0.01):
            ref = famp.get_ref(t)
            xs.append(ref.pos[0,0])
            ys.append(ref.pos[1,0])
        plt.plot(xs, ys)
        states.append(famp.get_ref(T))

    if check_states(states, correct['states']) == 3:
        print('Forward arc motion primitives correct')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.gca().set_aspect('equal')
        plt.show()
    else:
        print('Forward arc motion primitives failure')

def test_coeffs_bodyf2worldf():
    ''' Tis is an optional test function you can call within the __init__ funtion in forwardarctrajectory.py
    to convert coefficients from body frame to world frame '''
    coeffs_body_frame = np.array([
    [0.00000000e+00, 1.00000000e+00, -3.86738335e-14, -2.84217094e-14,
     -9.27967183e-15, -2.26559351e-01, 1.89474081e-01, -5.91032049e-02,
      6.64825537e-03],
    [0.00000000e+00, 8.88178420e-16, 1.07145339e-14, -8.41451632e-30,
     -1.00952294e-16, 4.40210510e-01, -4.34271402e-01, 1.52471142e-01,
     -1.87836971e-02],
    [0.00000000e+00, 0.00000000e+00, 2.01320442e-15, 0.00000000e+00,
     -2.96059473e-17, 8.75000000e-02, -8.75000000e-02, 3.12500000e-02,
     -3.90625000e-03],
    [0.00000000e+00, 8.88178420e-16, 1.00660221e-14, 0.00000000e+00,
     -1.48029737e-16, 4.37500000e-01, -4.37500000e-01, 1.56250000e-01,
     -1.95312500e-02]
    ])

    coeffs_world_frame = np.array([
        [1.00000000e+00, 1.00000000e+00, -3.86738335e-14, -2.84217094e-14,
        -9.27967183e-15, -2.26559351e-01, 1.89474081e-01, -5.91032049e-02,
          6.64825537e-03],
        [1.58603289e-15, 1.77635684e-15, 1.07145339e-14, -3.36580653e-29,
        -1.00952294e-16, 4.40210510e-01, -4.34271402e-01, 1.52471142e-01,
        -1.87836971e-02],
        [0.00000000e+00, 0.00000000e+00, 2.01320442e-15, 0.00000000e+00,
        -2.96059473e-17, 8.75000000e-02, -8.75000000e-02, 3.12500000e-02,
        -3.90625000e-03],
        [8.88178420e-16, 8.88178420e-16, 1.00660221e-14, 0.00000000e+00,
        -1.48029737e-16, 4.37500000e-01, -4.37500000e-01, 1.56250000e-01,
        -1.95312500e-02]
    ])
    
    curr_ref_position = np.array([1.00000000e+00, 1.58603289e-15, 0.00000000e+00])

    curr_ref_rotation = np.array([
        [1.0000000e+00, -8.8817842e-16,  0.0000000e+00],
        [8.8817842e-16,  1.0000000e+00,  0.0000000e+00],
        [-0.0000000e+00,  0.0000000e+00,  1.0000000e+00]
    ])

    Twb0 = Pose()
    Twb0.set_translation(curr_ref_position.flatten())
    Twb0.set_rotation(curr_ref_rotation)

    curr_ref = State()
    cmd = np.zeros((3,1))
    T = 2
    famp = ForwardArcTrajectory(curr_ref, cmd, T)
    cal_coeffs_world_frame = famp.coeffs_bodyf2worldf(coeffs_body_frame, Twb0)
    if(np.max(abs((coeffs_world_frame - cal_coeffs_world_frame))) < 1e-01):
        print("test_coeffs_bodyf2worldf passed! (OPTIONAL TEST)")
    else: 
        print("test_coeffs_bodyf2worldf failed! (OPTIONAL TEST)")

test_get_coefficients()
test_integrate_coefficients()
test_calculate_coefficients_from_constraints()
test_coeffs_bodyf2worldf()
test_forwardarcprimitives()
