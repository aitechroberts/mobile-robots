# 16-761 Assignment 0: Rotations and Transforms

Goals: In this assignment, you will implement the mathematics required
for converting between different rotation representations and
transforms. The code developed in this assignment will be required for
future assignments, so please start early.

### Academic Integrity
1. Do not publicly share your solution (using GitHub or otherwise)
2. Collaboration is encouraged but you should write final code on your own.
3. No AI tools may be used to complete this assignment. This includes
but is not limited to Copilot, ChatGPT, Perplexity AI, and Cursor AI.

### 0.0 Setup
Create a python virtual environment.
```python
python3.8 -m venv .venv
```
Python 3.6 -- 3.10 have all been tested and confirmed to work (Autolab runs 3.6). Source the environment
```python
source .venv/bin/activate
```
You will need to install the following dependencies.
```python
pip install numpy
```
Download the assignment.
```bash
git clone git@github.com:mr-cmu/assignment0-handout.git
```
Note: if the above command results in a `Permission denied (public key)`
error, then try setting up an SSH key in your Github account using the
instructions [here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account).

If the cloning still does not work, use
```bash
git clone https://github.com/mr-cmu/assignment0-handout.git
```

## 1.0 Problems (100 points)
You will need to edit the files `rotation3.py`, `pose.py`, and
`transforms.py`.  You will write the code to perform the following
conversions:

* Rotation matrix (3x3) -> Euler ZYX angles: `to_euler_zyx`
* Euler ZYX -> Rotation matrix (3x3): `from_euler_zyx`
* Rotation matrix (3x3) -> Quaternion (w,x,y,z): `to_quat`
* Quaternion (w,x,y,z) -> Rotation matrix (3x3): `from_quat`
* Calculate roll, pitch, and yaw
* Take composition and inverse of 4x4 transformations

More information about each function follows.  Your code will be
graded using Autolab. See Section 2 for details about uploading and
receiving scores for your implementations.

### 1.1 `to_euler_zyx`
This function calculates the angles `phi=X`, `theta=Y`, `psi=Z` that
represent the rotation in the Z-Y-X Tait-Bryant
parameterization. The expected output is a 1x3 numpy array.  Note: the
expected output is in reverse order from functions like MATLAB's
`rotm2eul`; however, you can use this function to check your results.

### 1.2 `from_euler_zyx`
This function calculates the 3x3 rotation matrix from the input angles
`phi=X`, `theta=Y`, and `psi=Z`.

### 1.3 `roll`
This function extracts and returns the phi component from the
rotation matrix.

### 1.4 `pitch`
This function extracts and returns the theta component from the
rotation matrix.

### 1.5 `yaw`
This function extracts and returns the psi component from the
rotation matrix.

### 1.6 `from_quat`
This function calculates the 3x3 rotation matrix from a
(w,x,y,z)-parameterized quaternion.

### 1.7 `to_quat`
This function calculates the (w,x,y,z) quaternion from a 3x3 rotation
matrix.

### 1.8 `compose`
This function composes a 4x4 transform represented in the `Pose`
object with another `Pose` object. This is found in the `pose.py`
file.

### 1.9 `inverse`
This function takes the inverse of a 4x4 transform represented in the
`Pose` object. This is found in the `pose.py` file.

### 1.10 `set_robot_transforms`
In this function you will find the transform that represents the pose
of the camra in the world frame. This function may be found in the
`transforms.py` file.

## Checking your results
There is a test in `test/tests.py`. It runs exactly the same tests as
what is on Autolab for ease of debugging. You will receive information
about whether or not your implementation passes each test.

## 2. Grading with AutoLab
To have your solutions graded, you will need to tar the `quadrotor_simulator_py`
folder and upload to autolab.

```bash
cd assignment0-handout
tar -cvf handin.tar quadrotor_simulator_py
```

Autolab will run tests on each function you implement and you will
receive a score out of 100.  You may upload as many times as you like.
Note that we may regrade submissions after the deadline passes.