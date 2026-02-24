# Quadrotor Control Architecture: Implementation Notes & Bug Fixes

This document tracks critical discrepancies between analytical quadrotor dynamics and computational implementations, specifically addressing numerical stability and integration state management.

## 1. ODE Integrator State Management (Quaternion Derivative)

**The Issue:**
When numerically integrating the robot dynamics over time, the raw state vector must be passed directly to the derivative calculations to maintain internal consistency within the Runge-Kutta solver. Normalizing the quaternion *before* calculating its derivative artificially scales the rate of change. This disrupts the ODE solver's evaluation steps, causing integration drift and cascading state tracking failures.

**Incorrect Implementation:**
* **Mathematical Concept:** $\dot{q} = \frac{1}{2} \frac{q}{||q||} \otimes \begin{bmatrix} 0 \\ \omega_b \end{bmatrix}$
* **Code:**
    ```python
    qn = Quaternion(quat).normalize()
    dq = self.quaternion_derivative(qn, wb)
    ```

**Correct Implementation:**
[cite_start]The derivative of a quaternion must be calculated using its current raw magnitude[cite: 20]. Normalize the quaternion only when extracting the rotation matrix $R_{wb}$, but retain the raw unnormalized quaternion for the $\dot{q}$ calculation.
* [cite_start]**Mathematical Concept:** $\dot{q}=\frac{1}{2}q\otimes[\begin{matrix}0\\ \omega_{b}\end{matrix}]$ [cite: 20]
* **Code:**
    ```python
    qn = Quaternion(quat).normalize() # Used ONLY for Rwb extraction
    q_raw = Quaternion(quat)
    dq = self.quaternion_derivative(q_raw, wb)
    ```

---

## 2. Rotation Matrix Orthogonalization (Floating-Point Precision)



**The Issue:**
When constructing the desired rotation matrix $R_{des} = [x_B, y_B, z_B]$, $x_B$ is built to be orthogonal to $z_B$. Analytically, the cross product of two orthogonal unit vectors ($z_B \times x_B$) inherently results in a perfect unit vector. Computationally, floating-point precision loss causes the resulting vector to microscopically drift away from a magnitude of 1.0. Failing to normalize this final basis vector violates the strict properties of the rotation matrix, causing severe Root Mean Square Error (RMSE) accumulation in the cascaded position and attitude controllers.

**Incorrect Implementation:**
[cite_start]While shorthand lecture slides often omit the denominator for brevity[cite: 349], direct computational implementation of the shorthand causes numerical drift.
* [cite_start]**Mathematical Concept:** $y_{B,des}=z_{B,des}\times x_{B,des}$ [cite: 349]
* **Code:**
    ```python
    y_B = np.cross(z_B, x_B, axis=0)
    ```

**Correct Implementation:**
[cite_start]Formal differential flatness proofs explicitly enforce the normalization of the final basis vector to maintain proper coordinate frame transformations[cite: 583].
* [cite_start]**Mathematical Concept:** $y_{B}=\frac{z_{B}\times x_{B}}{||z_{B}\times x_{B}||}$ [cite: 583]
* **Code:**
    ```python
    y_B_raw = np.cross(z_B, x_B, axis=0)
    y_B = y_B_raw / norm(y_B_raw)
    ```