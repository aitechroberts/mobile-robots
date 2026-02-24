# Useful Equations Reference

Quick-reference for the math behind the quadrotor simulator, position controller,
and attitude controller. Written in plain English with the actual equations used in code.

---

## 1. Quadrotor Dynamics (Q1)

### Motor Model

Each rotor produces thrust as a quadratic function of its RPM:

    f_i = cT2 * rpm_i^2 + cT1 * rpm_i + cT0

The coefficients come from the YAML config. cT2 is tiny (~4e-8), cT1 is small negative,
and cT0 is a constant offset (~0.19 N).

### Mixer Matrix

The mixer converts four individual rotor forces into a total thrust and three torques:

    [thrust, tau_x, tau_y, tau_z]^T = Mixer @ [f1, f2, f3, f4]^T

For a QuadX layout with arm length L and motor spread angle a (pi/4):
- Row 0: thrust = f1 + f2 + f3 + f4
- Row 1: roll torque, with moment arm L*sin(a)
- Row 2: pitch torque, with moment arm L*cos(a)
- Row 3: yaw torque, with reaction torque scale k

### Quaternion Derivative

    dq/dt = (1/2) * Omega(w) * q

where q = [w, x, y, z] and Omega is a 4x4 matrix built from the body angular velocity.
**Important:** use the raw (unnormalized) quaternion for the derivative so the ODE
integrator stays consistent. Only normalize when converting to a rotation matrix.

### World-Frame Linear Acceleration (Mellinger Eq 4.2)

    a_world = -g*e3 + (F/m)*R*e3 + R*(r_off x alpha) + R*(w x (w x r_off))

- `-g*e3`: gravity pulling down
- `(F/m)*R*e3`: thrust along body-z rotated to world frame
- `R*(r_off x alpha)`: tangential acceleration from CG offset
- `R*(w x (w x r_off))`: centripetal acceleration from CG offset

When the CG offset is zero (common case), the last two terms vanish.

### Body-Frame Angular Acceleration (Mellinger Eq 4.3)

    alpha = I^{-1} * (M - w x Iw - w x (r_off x F) + r_off x m*g*e3)

- `M`: torques from rotors
- `w x Iw`: gyroscopic (Coriolis) torque from spinning body
- Last two terms: CG offset corrections (zero when CG = geometric center)

### ODE Step -- Key Pitfall: Rotor Inertia Flag

When `enable_rotor_inertia = false`, motors respond **instantly**. The force/torque
calculation must use the **commanded** RPMs (`uRPM`), not the current state RPMs
which may be stale from the previous timestep. Getting this wrong causes a
one-step thrust lag that compounds into visible position drift over ~40 seconds.

---

## 2. Position Controller (Q2)

### Desired Acceleration (PD Feedback)

    a_des = a_ref + g*e3 - Kp*(p - p_ref) - Kd*(v - v_ref)

This is the acceleration vector the quadrotor needs to achieve. It includes:
- `a_ref`: feedforward from the reference trajectory
- `g*e3`: gravity compensation (so hover requires zero feedback)
- PD terms: proportional on position error, derivative on velocity error

### Body-Z Acceleration (Thrust Command)

    c = a_des . (R_curr * e3)

Projects the desired acceleration onto the **current** body z-axis. This gives
the mass-normalized scalar thrust the motors need to produce.

### Desired Orientation (Differential Flatness)

    z_B = a_des / ||a_des||           (thrust direction = body z)
    x_C = [cos(psi), sin(psi), 0]     (heading reference)
    y_B = (z_B x x_C) / norm          (body y: perpendicular to thrust & heading)
    x_B = y_B x z_B                   (body x: completes right-handed frame)
    R_des = [x_B, y_B, z_B]

The key insight: the direction of thrust uniquely determines the tilt of the
quadrotor. The yaw reference then fixes the remaining rotational degree of freedom.

### Angular Velocities (Daoud/Goel/Tabib Sec VI-B)

From the jerk (derivative of acceleration), we get body angular velocities:

    omega_x = -(y_B . jerk) / c       (roll rate)
    omega_y =  (x_B . jerk) / c       (pitch rate)

For omega_z (yaw rate), there is a **critical geometric coupling** (Eq 22):

    omega_z = (psi_dot * xc.xB + omega_y * yc.zB) / ||yc x zB||

The `omega_y * yc.zB` term is the key: even with zero yaw rate (psi_dot = 0),
pitching while tilted causes body yaw. This is a geometric effect -- when z_B is
not vertical, tilting the body also rotates the heading projection.

### Angular Accelerations (Daoud/Goel/Tabib Sec VI-C)

From the snap (derivative of jerk), projecting onto body axes:

    alpha_y = (xB.snap - 2*c_dot*wy - c*wx*wz) / c
    alpha_x = (-yB.snap - 2*c_dot*wx + c*wy*wz) / c

The `2*c_dot*w` terms come from differentiating the product `c * z_B_dot`.
The `w_i*w_z` cross-terms come from centripetal coupling in the rotating frame.

alpha_z is the most complex (derived by differentiating the omega_z formula):

    alpha_z = (1/||yc x zB||) * (
        d2yaw * xc.xB
        + dyaw * wz * xc.yB
        - 2 * dyaw * wy * xc.zB
        + alpha_y * yc.zB
        - wx * wy * yc.yB
        - wz * (-dyaw * xc.yB + wx * yc.zB)
    )

Even at zero yaw rate, the `alpha_y * yc.zB` and `wx*wy * yc.yB` terms produce
nonzero body yaw acceleration through the tilt-yaw geometric coupling.

---

## 3. Attitude Controller (Q3)

### Rotation Error (Mellinger p.21)

    eR = (1/2) * vee(R_des^T * R - R^T * R_des)

The vee map extracts a 3-vector from a skew-symmetric matrix:

    vee([[0,-c,b],[c,0,-a],[-b,a,0]]) = [a, b, c]

This error metric is zero when R = R_des and grows proportionally to the
rotation angle for small errors. It works globally (no gimbal lock).

### Angular Velocity Error

    eOm = omega - R^T * R_des * omega_des

The `R^T * R_des` maps the desired angular velocity from the desired body frame
into the current body frame before computing the difference.

### Desired Moments (Spitzer Eq 2.68 x Inertia)

    alpha_cmd = alpha_des - kR * eR - kOm * eOm     (PD on SO(3))
    M = I * alpha_cmd + omega x (I * omega)          (Euler's equation)

- `I * alpha_cmd`: torque needed to achieve the desired angular acceleration
- `omega x (I * omega)`: compensates for gyroscopic precession (the tendency of
  a spinning body to resist changes in orientation)

### Wrench to Rotor Forces

    rotor_forces = Mixer^{-1} * [thrust; torque_x; torque_y; torque_z]

### Force to RPM (Inverse Motor Model)

    cT2 * rpm^2 + cT1 * rpm + (cT0 - f) = 0

Solved with the quadratic formula, taking the positive root:

    rpm = (-cT1 + sqrt(cT1^2 - 4*cT2*(cT0 - f))) / (2*cT2)

---

## Notation Quick Reference

| Symbol | Meaning |
|--------|---------|
| R, R_des | Current / desired rotation matrix (body-to-world) |
| omega, w | Body-frame angular velocity [wx, wy, wz] |
| alpha | Body-frame angular acceleration |
| e3 | Unit vector [0, 0, 1] (world up / body z) |
| c | Mass-normalized thrust = ||a_des|| |
| j, s | Jerk and snap (3rd/4th derivatives of position) |
| xc, yc | Heading vectors: [cos(psi), sin(psi), 0] and [-sin(psi), cos(psi), 0] |
| xB, yB, zB | Body frame axes (columns of R_des) |
| I | 3x3 inertia matrix |
| kR, kOm | Diagonal gain matrices for rotation / angular velocity PD |
| eR, eOm | Rotation error / angular velocity error |
| r_off | CG offset from geometric center (usually zero) |

---

## References

- [1] Mellinger, "Trajectory Generation and Control for Quadrotors" (PhD thesis, 2012)
- [2] Faessler et al., "Differential Flatness of Quadrotor Dynamics Subject to Rotor Drag" (RAL 2018)
- [3] Daoud, Goel, Tabib, "Differential Flatness of Quadrotor Dynamics under General Conditions using the ZYX Euler Convention"
- [4] Spitzer, "Dynamical Model Learning and Inversion for Aggressive Quadrotor Flight" (CMU-RI-TR-22-03)
