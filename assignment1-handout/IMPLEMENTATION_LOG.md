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

**Bug #2 found and fixed (motor dynamics with rotor inertia disabled):**
- **Symptom:** Position still drifted even after quaternion fix. Student solution
  diverged smoothly from correct solution starting ~25-30s, growing to several
  meters of error by 60s. Shape of curves matched but was offset.
- **Root cause:** When `enable_rotor_inertia = False`, the `update` method
  overrides `self.rs = self.model_params.uRPM` after each integration step.
  On the next step, `x[13:17]` (RPM state) starts at the PREVIOUS command's
  RPMs. The force calculation `calculate_force_and_torque_from_rpm(rpms)` used
  these stale RPMs instead of the current commanded RPMs. This created a
  systematic one-step thrust lag: forces during each integration interval were
  based on the previous command, not the current one. The small force errors
  compounded into visible position drift over time.
- **Fix:** When `enable_rotor_inertia = False`, use the commanded RPMs
  (`self.model_params.uRPM`) directly for force/torque calculation, bypassing
  the ODE RPM state. This models ideal (instantaneous) motor response:
  ```python
  if self.enable_rotor_inertia:
      rpms_for_force = rpms        # use ODE state (motor dynamics active)
  else:
      rpms_for_force = uRPM        # use commanded RPMs directly (ideal motors)
  F, M = self.calculate_force_and_torque_from_rpm(rpms_for_force)
  ```
- **Why unit tests didn't catch it:** The individual function tests (1.2--1.5)
  don't involve the ODE loop. The motor dynamics only matter during time
  integration where RPM state evolves from one command to the next.

---

## Q2: Position Controller

### 2.1 `compute_body_z_accel` (positionpd.py) -- PASS
Projection of desired acceleration onto current body z-axis:
`u = a_des^T @ R_curr @ e3`

### 2.2 `compute_orientation` (positionpd.py) -- PASS
Faessler Eq (33)-(36): construct R_des from desired acceleration and yaw reference.
`z_B = a_des / ||a_des||`, then `y_B = (z_B x x_C) / norm`, `x_B = y_B x z_B`.

### 2.3 `compute_hod_refs` (positionpd.py) -- PASS (after rewrite)

**Initial attempt (FAILED):** Used simplified formulas from plan:
- `omega_z = dyaw * (e3 · z_B)` and `alpha_z = d2yaw * (e3 · z_B)`
- `h_alpha` used `||h_w||^2 * a_des` instead of correct `Tddot * z_B` term

**Bug: Geometric tilt-yaw coupling missing.**
- **Symptom:** Angular velocity x,y components matched perfectly, but z-component
  was nearly zero. Correct answer had ω_z = 0.00922 even with dyaw ≈ 0.
  Angular acceleration z-component was similarly flat vs oscillating correct.
- **Root cause:** When the body tilts (z_B changes direction), the heading
  (projection of x_B onto the horizontal plane) rotates even without any yaw
  rate. The simplified formula `ω_z = dyaw * (e3 · z_B)` completely missed
  this geometric coupling. Similarly, the angular acceleration formula was
  missing both this coupling and cross-terms (q*r, -p*r).
- **Fix:** Replaced the simplified projection formulas with full rotation
  matrix kinematics:
  1. Compute `dR/dt = [dx_B/dt, dy_B/dt, dz_B/dt]` from flatness outputs
     by differentiating `z_B`, `y_B = (z_B × x_C)/norm`, `x_B = y_B × z_B`
  2. Extract ω from `vee(R^T @ dR/dt)` (captures all couplings)
  3. Compute `d²R/dt²` by differentiating dR/dt again
  4. Extract α from `vee(skew(R^T @ d²R/dt²))` (captures all cross-terms)
- **Also fixed h_alpha:** Corrected `Tddot = h_w · jerk + z_B · snap`
  (was incorrectly using `||h_w||^2 * a_des`)

### 2.4 `compute_command` (positionpd.py)
PD feedback + feedforward + gravity:
`a_des = acc_ref + g*e3 - Kx*e_pos - Kv*e_vel`
Then calls compute_orientation, compute_body_z_accel, compute_hod_refs.

---

## Q3: Attitude Controller

*(not yet implemented)*
