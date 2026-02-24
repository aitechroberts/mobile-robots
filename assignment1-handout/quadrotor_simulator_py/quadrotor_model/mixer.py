#!/usr/bin/env python

import numpy as np
import yaml


class QuadMixer:

    def __init__(self):
        self.mixer = np.zeros((4, 4))
        self.length = 0.0
        self.mscale = 0.0
        self.ms_angle = 0.0

    def __repr__(self):
        return ('QuadMixer\n' +
                'mixer:\n' + np.array2string(self.mixer) + '\n' +
                'length: ' + str(self.length) + '\n' +
                'mscale: ' + str(self.mscale) + '\n' +
                'ms_angle: ' + str(self.ms_angle) + '\n'
                )

    def initialize(self, yaml_file):
        data = []
        with open(yaml_file, 'r') as stream:
            try:
                data = yaml.safe_load(stream)
            except yaml.YamlError as exc:
                print(exc)

        self.length = data['length']
        self.mscale = data['rotor']['moment_scale']
        self.ms_angle = data['motor_spread_angle']

    def construct_mixer(self):

        """ Calculates the mixer matrix, which converts
                rotor speeds (e.g., RPMs) to force and torques.
                Assumes NWU.

        Output:
            mixer: 4x4 numpy matrix
        """

        L = self.length
        k = self.mscale
        a = self.ms_angle

        # Mixer maps individual rotor forces [f1,f2,f3,f4] to [thrust, tau_x, tau_y, tau_z].
        # QuadX layout with motor_spread_angle a (pi/4 for standard X config):
        #   Row 0: total thrust = sum of all rotor forces
        #   Row 1: roll torque  = moment arm L*sin(a) with alternating signs per arm
        #   Row 2: pitch torque = moment arm L*cos(a) with alternating signs per arm
        #   Row 3: yaw torque   = reaction torques scaled by moment_scale k
        self.mixer = np.array([
            [1.0,          1.0,          1.0,          1.0         ],
            [-L*np.sin(a), L*np.sin(a),  L*np.sin(a), -L*np.sin(a)],
            [-L*np.cos(a), L*np.cos(a), -L*np.cos(a),  L*np.cos(a)],
            [-k,          -k,            k,            k           ]
        ])

        return self.mixer
