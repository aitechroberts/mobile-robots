# Quadrotor ODE Integration: RPM Motor Dynamics Bug & Analysis

This document archives the definitive analysis of the `ode_step` integration failure in the quadrotor simulation, specifically detailing how an inverted motor dynamics equation cascaded into massive angular acceleration errors, along with other critical numerical stability fixes.

## 1. The Core Issue: First-Order Motor Dynamics

The fundamental cause of the integration failure was a single algebraic inversion in the first-order motor model. 

The physical equation for a motor tracking a commanded RPM is defined as a first-order low-pass filter:
$$\dot{\omega} = \frac{\omega_{cmd} - \omega}{\tau}$$
Where $\tau$ is the motor's time constant (the time it takes to reach ~63.2% of the target step command).

**Incorrect Implementation (Multiplication):**
```python
# The YAML field maps 'time_constant' to 'kmotor_u'
kmotor = self.model_params.kmotor_u
drpm = kmotor * (uRPM - rpms)  # BUG: Multiplying by the time constant

# Quadrotor Simulation Debugging Report

## 1. Primary Failure: Motor Time Constant Logic
For a typical quadrotor, the motor time constant $\tau$ is approximately 0.05 seconds. The simulation failed the autograder due to a mathematical inversion in the motor dynamics: 

* **The Bug:** The code multiplied by $\tau$ instead of dividing. 
* **The Result:** Simulated motors were given a gain of $0.05$ instead of the required $1 / 0.05 = 20.0$. This effectively "froze" the motors, preventing them from reaching commanded speeds and generating incorrect forces/moments.

### The Config Masking Effect
Local testing passed because the `rocky0704_model_params.yaml` contained a physically impossible parameter:
`time_constant: 18.6108`

Since $\frac{1}{18.6108} \approx 0.0537$, the local configuration was "pre-inverted" to cancel out the buggy multiplication logic. When the autograder applied a pristine config with a true $\tau$ (0.0537), the logic failed catastrophically.

---

## 2. Error Signature Analysis (The Smoking Gun)
The mathematical fingerprint of this bug was identified via the **Root Mean Square Error (RMSE)** cascade.

| State Metric | RMSE | Error Magnitude vs Threshold | Type |
| :--- | :--- | :--- | :--- |
| Angular Acceleration | 0.113 | 113x threshold | Derivative |
| Linear Acceleration | 0.039 | 39x threshold | Derivative |
| Orientation | 0.021 - 0.047 | 21x - 47x threshold | Integral |
| Position | 0.003 | 3x threshold | Double Integral |
| Velocity | 0.002 | 2x threshold | Integral |

**Inference:**
Derivative quantities (accelerations) exhibited the highest magnitude of error. Because angular acceleration $\alpha$ is derived strictly from RPM-based moments, the fact that $\alpha$ was the most broken metric isolated the bug to the **RPM derivative generation**.

---

## 3. Secondary Technical Resolutions

### A. Quaternion Normalization
In the `ode_step`, the raw state vector quaternion is no longer normalized prior to calculating the derivative $\dot{q} = \frac{1}{2} q \otimes \begin{bmatrix} 0 \\ \omega_b \end{bmatrix}$. This ensures mathematical rigor, as artificial normalization scales the derivative by $\frac{1}{||q||}$. Normalization is now strictly reserved for extracting the Rotation Matrix $R_{wb}$.

### B. Tracking Error Angle Wrap
Corrected an algebraic error in `geometry.py`:
```python
# Fixed: logic now correctly shifts the wrap by 2*pi
result -= 2 * math.pi

### C. Runge-Kutta Side Effects & Tolerances
By default, `scipy.integrate.solve_ivp` probes the derivative at intermediate Runge-Kutta stages (e.g., $t + \frac{1}{2}dt$). If class variables (like `self.model_params.aw`) are mutated inside `ode_step`, they retain the values of the last intermediate probe, not the final integrated state. 

* **The Fix:** Forced one final evaluation of `ode_step` using the exact final state $t_{stop}$ to lock in the correct accelerations. 
* **Precision:** Added strict bounds (`rtol=1e-6`, `atol=1e-8`) to eliminate step-boundary truncation drift.

### D. Saturation Indexing Bug
In `model.py`, the `_saturate_rpms` function contained a subtle indexing flaw where a manual loop evaluated `rpm_in[0]` for all four rotor iterations. This was resolved by utilizing `np.clip` to handle element-wise array bounding:

```python
def _saturate_rpms(self, p, rpm_in):
    return np.clip(rpm_in, p.min, p.max)

    