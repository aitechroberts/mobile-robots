# Assignment 1 -- Implementation Log

## Q1: Quadrotor Simulator

### 1.1 `construct_mixer` (mixer.py) -- PASS
Built the 4x4 NWU QuadX mixer matrix mapping rotor forces to [F, tau_x, tau_y, tau_z].
Uses arm length L, moment scale kappa, and motor spread angle alpha.

### 1.2 `calculate_force_and_torque_from_rpm` (model.py) -- PASS
Per-rotor force from quadratic motor model (cT2*rpm^2 + cT1*rpm + cT0),
then multiply by mixer to get total force and 3x1 torque vector.

### 1.3 `quaternion_derivative` (model.py) -- PASS
Standard formula: dq/dt = 0.5 * Omega(omega) @ q, where Omega is the
4x4 skew-symmetric-like matrix built from body angular velocities.

### 1.4 `calculate_world_frame_linear_acceleration` (model.py) -- PASS
Mellinger Eq (4.2): a_w = -g*e3 + (u1/m)*R*e3 + R*(r_off x ang_acc) + R*(wb x (wb x r_off)).
Includes full CG offset terms for generality (zeroed in current config).

### 1.5 `calculate_angular_acceleration` (model.py) -- PASS
Mellinger Eq (4.3): ang_acc = I_inv @ (M - wb x I*wb - wb x (r_off x Fdes) + r_off x [0,0,mg]).
Includes full CG offset terms for generality (zeroed in current config).

### 1.6 `ode_step` (model.py) -- PASS (after fix)
Assembles the full 17-state ODE: [vel, dq, aw, ang_acc, drpm].

**Bug found and fixed:**
- **Symptom:** Position plot matched perfectly until ~40 seconds, then drifted
  with a compounding error. All other plots (velocity, acceleration, angular
  velocity, angular acceleration) appeared correct.
- **Root cause:** Quaternion was normalized before computing *both* the rotation
  matrix and the quaternion derivative. The ODE solver integrates the raw
  (unnormalized) quaternion state. Using the normalized quaternion in the
  derivative formula scales the derivative by 1/||q||, breaking the
  skew-symmetric norm-preservation property (q^T @ Omega @ q = 0) and giving
  the solver an inconsistent derivative for the state it's actually integrating.
  Over time, the small rotation error caused a tiny thrust-direction offset that
  integrated into visible position drift.
- **Fix:** Use the raw quaternion for the derivative, normalized quaternion only
  for computing the rotation matrix:
  ```python
  qn_raw = Quaternion(quat)               # for derivative
  qn_norm = Quaternion(quat).normalize()   # for rotation matrix
  Rwb = Rot3.from_quat(qn_norm).R
  # ...
  dq = self.quaternion_derivative(qn_raw, wb)
  ```
- **Why unit test didn't catch it:** The `test_quaternion_derivative` test
  uses a quaternion with ||q|| = 1.0 exactly, so normalizing had no effect.
  The bug only manifests during long-horizon ODE integration where quaternion
  norm drifts from 1.0.

---

## Q2: Position Controller

*(not yet implemented)*

## Q3: Attitude Controller

*(not yet implemented)*
