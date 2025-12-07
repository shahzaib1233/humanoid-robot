---
title: Theoretical Foundations Exercises
sidebar_label: Exercises
sidebar_position: 4
description: Practice problems for the theoretical foundations chapter
keywords: [exercises, problems, practice, control theory, mathematics, robotics]
---

# Theoretical Foundations Exercises

These exercises reinforce the mathematical and theoretical concepts covered in the Theoretical Foundations chapter. Work through these problems to deepen your understanding of the fundamental principles underlying Physical AI and humanoid robotics.

## Exercise 1: Forward Kinematics for a 2-DOF Planar Manipulator

**Difficulty**: Basic

**Estimated Time**: 30-45 minutes

### Problem
Consider a 2-degree-of-freedom (DOF) planar manipulator with link lengths l₁ = 0.5m and l₂ = 0.4m. The joint angles are θ₁ = π/4 radians and θ₂ = π/6 radians. Calculate the position (x, y) of the end-effector in the base coordinate frame.

### Instructions
1. Draw a diagram of the manipulator showing the links and joint angles
2. Use trigonometry to derive the forward kinematics equations
3. Substitute the given values to find the end-effector position
4. Verify your result using geometric reasoning

### Expected Output
- Forward kinematics equations for x and y in terms of θ₁, θ₂, l₁, and l₂
- Numerical values for the end-effector position
- Diagram showing the manipulator configuration

### Solution Approach
Use the geometric relationships between joint angles and link positions to derive the transformation equations.

### Hints
- Consider the position of the elbow joint first
- Add the second link's contribution to get the end-effector position
- Use trigonometric identities as needed

---

## Exercise 2: Jacobian Matrix Calculation

**Difficulty**: Intermediate

**Estimated Time**: 45-60 minutes

### Problem
For the 2-DOF planar manipulator from Exercise 1, derive the Jacobian matrix that relates joint velocities (θ̇₁, θ̇₂) to end-effector velocities (ẋ, ẏ). Then, if joint velocities are θ̇₁ = 0.2 rad/s and θ̇₂ = -0.1 rad/s, calculate the resulting end-effector velocity.

### Instructions
1. Start with the forward kinematics equations from Exercise 1
2. Take partial derivatives with respect to each joint angle
3. Form the Jacobian matrix J = [∂x/∂θ₁  ∂x/∂θ₂]
                               [∂y/∂θ₁  ∂y/∂θ₂]
4. Calculate the end-effector velocity using v = J * θ̇

### Expected Output
- Complete derivation of the Jacobian matrix
- Numerical evaluation of the Jacobian for the given configuration
- End-effector velocity vector

### Solution Approach
Use the definition of the Jacobian as the matrix of partial derivatives of the forward kinematics equations.

### Hints
- The Jacobian maps joint space velocities to Cartesian space velocities
- Remember to use the chain rule when taking derivatives
- Check that your units are consistent

---

## Exercise 3: PID Controller Design

**Difficulty**: Intermediate

**Estimated Time**: 60-90 minutes

### Problem
Design a PID controller for a simple mass-spring-damper system with parameters: mass m = 2 kg, damping coefficient c = 1 Ns/m, and spring constant k = 10 N/m. The system should track a step reference input with minimal overshoot and a settling time of approximately 4 seconds.

### Instructions
1. Derive the transfer function of the mass-spring-damper system
2. Analyze the open-loop system response
3. Design PID gains using a suitable method (Ziegler-Nichols, pole placement, or optimization)
4. Simulate the closed-loop system response
5. Evaluate performance metrics (rise time, settling time, overshoot, steady-state error)

### Expected Output
- System transfer function
- Justification for PID gain selection
- Simulation results showing step response
- Performance metrics evaluation

### Solution Approach
Use control theory principles to design the controller, considering both stability and performance requirements.

### Hints
- The characteristic equation for the mass-spring-damper system is ms² + cs + k = 0
- For a settling time of 4 seconds, consider the relationship between settling time and system poles
- Use the final value theorem to check steady-state error

---

## Exercise 4: Stability Analysis Using Lyapunov Method

**Difficulty**: Advanced

**Estimated Time**: 90-120 minutes

### Problem
Analyze the stability of the following nonlinear system using Lyapunov's direct method:
```
ẋ₁ = x₂
ẋ₂ = -x₁ - x₂³
```

The equilibrium point is at (0, 0). Propose a Lyapunov function candidate and prove the stability of the system.

### Instructions
1. Identify the equilibrium point(s) of the system
2. Propose a suitable Lyapunov function candidate V(x)
3. Verify that V(0) = 0 and V(x) > 0 for x ≠ 0
4. Calculate the time derivative V̇(x) along system trajectories
5. Determine the stability properties based on V̇(x)

### Expected Output
- Verification that (0, 0) is an equilibrium point
- Proposed Lyapunov function with justification
- Analysis of V(x) properties
- Calculation of V̇(x) and stability conclusion

### Solution Approach
Consider energy-based Lyapunov functions for mechanical systems, or quadratic forms for general stability analysis.

### Hints
- A common choice for mechanical systems is V(x) = ½x₁² + ½x₂² (kinetic + potential energy)
- For the time derivative, use the chain rule: V̇ = (∂V/∂x₁)ẋ₁ + (∂V/∂x₂)ẋ₂
- Consider the sign definiteness of V̇(x)

---

## Exercise 5: Control of a Simple Inverted Pendulum

**Difficulty**: Advanced

**Estimated Time**: 120-150 minutes

### Problem
Design a state-feedback controller for an inverted pendulum on a cart. The linearized system equations are:
```
[ ẋ ]     [ 0    0   1   0 ][ x ]   [     0    ]
[ θ̇ ]  =  [ 0    0   0   1 ][ θ ] + [     0    ] * u
[ẍ ]     [ 0  g/l  0   0 ][ ẋ]   [ 1/(M+m) ]
[θ̈ ]     [ 0   g   0   0 ][θ̇]   [   1/ml   ]
```

Where x is cart position, θ is pendulum angle, M is cart mass, m is pendulum mass, l is pendulum length, and g is gravity. Use the following parameters: M = 0.5 kg, m = 0.2 kg, l = 0.3 m, g = 9.81 m/s².

### Instructions
1. Verify the linearized system equations
2. Check the controllability of the system
3. Design a state-feedback controller u = -Kx using pole placement or LQR
4. Choose appropriate closed-loop pole locations or weighting matrices
5. Simulate the closed-loop system response to an initial condition

### Expected Output
- Controllability matrix and verification of controllability
- Controller gain matrix K
- Justification for pole locations or weighting matrices
- Simulation results showing stabilization

### Solution Approach
Use linear control theory to design a stabilizing controller for the linearized system.

### Hints
- The controllability matrix is C = [B AB A²B A³B] for a 4th order system
- For pole placement, choose poles in the left half-plane for stability
- For LQR, try Q = I (identity) and R = 1 initially, then adjust as needed

---

## Exercise 6: ZMP-Based Balance Control

**Difficulty**: Advanced

**Estimated Time**: 120-180 minutes

### Problem
Implement a Zero Moment Point (ZMP) based balance controller for a simplified humanoid robot model. The robot is modeled as a linear inverted pendulum with height h = 0.8 m. Design a controller that maintains the ZMP within a support polygon defined by the feet positions.

### Instructions
1. Derive the LIPM (Linear Inverted Pendulum Model) equations
2. Define the ZMP calculation for the inverted pendulum
3. Design a feedback controller to regulate ZMP position
4. Implement the controller in simulation
5. Test with disturbances and evaluate stability

### Expected Output
- LIPM equations derivation
- ZMP calculation equations
- Controller design with parameters
- Simulation results showing balance maintenance
- Analysis of stability margins

### Solution Approach
Use the relationship between center of mass motion and ZMP to design a stabilizing controller.

### Hints
- LIPM equation: ẍ_com = g/h * (x_com - x_zmp)
- The ZMP should remain within the convex hull of contact points for stability
- Consider using a preview controller that anticipates future ZMP positions

---

## Exercise 7: Kalman Filter Implementation

**Difficulty**: Intermediate

**Estimated Time**: 90-120 minutes

### Problem
Implement a Kalman filter to estimate the position and velocity of a moving robot given noisy position measurements. The system model is:
```
x_k = A * x_{k-1} + w_{k-1}
z_k = H * x_k + v_k
```

Where x_k = [position_k, velocity_k]ᵀ, A = [1 dt; 0 1], H = [1 0], with dt = 0.1s. Process noise covariance Q = [0.01 0; 0 0.01], measurement noise R = 0.1.

### Instructions
1. Implement the prediction step: x̂_k⁻ = A * x̂_{k-1}, P_k⁻ = A * P_{k-1} * Aᵀ + Q
2. Implement the update step: K_k = P_k⁻ * Hᵀ * (H * P_k⁻ * Hᵀ + R)⁻¹, x̂_k = x̂_k⁻ + K_k * (z_k - H * x̂_k⁻), P_k = (I - K_k * H) * P_k⁻
3. Test with simulated noisy measurements
4. Compare filtered estimates with true values

### Expected Output
- Complete Kalman filter implementation
- Simulation results showing filtering performance
- Comparison of noisy measurements vs. filtered estimates
- Analysis of filter convergence

### Solution Approach
Follow the standard Kalman filter algorithm with proper matrix operations.

### Hints
- Initialize with reasonable estimates for state and covariance
- The Kalman gain balances trust in predictions vs. measurements
- Monitor the error covariance to ensure filter stability

---

## Learning Objectives Addressed

- **Mathematical Modeling**: Apply linear algebra, calculus, and differential equations to robotics problems
- **System Analysis**: Analyze stability, controllability, and performance of robotic systems
- **Controller Design**: Design feedback controllers for various robotic applications
- **State Estimation**: Implement filters for handling sensor uncertainty
- **Advanced Control**: Apply modern control techniques to humanoid robotics problems

## Solutions Guide

Detailed solutions to these exercises are available to educators and verified students. The solutions include complete derivations, implementation code, and analysis of results. Contact your instructor or submit a request through the appropriate channels for access to detailed solutions.

## Additional Challenges

For advanced students, consider extending these exercises:

1. **Robustness Analysis**: Analyze how controllers perform with model uncertainties
2. **Optimization**: Use optimization techniques to tune controller parameters
3. **Multi-Robot Systems**: Extend single-robot control to multi-robot coordination
4. **Learning-Based Control**: Implement adaptive or learning-based control approaches
5. **Hardware Implementation**: Test controllers on physical robotic platforms

## References for Further Study

[This section will be expanded with proper academic citations in the final version]

## Tools and Software

For implementing and testing these exercises, consider using:
- MATLAB/Simulink for control system design and simulation
- Python with NumPy, SciPy, and Control Systems Library
- ROS/ROS2 with control packages
- Simulation environments like Gazebo or PyBullet
- Real robotic platforms for hardware validation