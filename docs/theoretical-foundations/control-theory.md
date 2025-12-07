---
title: Control Theory
sidebar_label: Control Theory
sidebar_position: 3
description: Fundamentals of control theory for humanoid robotics
keywords: [control theory, feedback control, stability, PID, robotics]
---

# Control Theory for Humanoid Robotics

Control theory provides the mathematical framework for designing algorithms that regulate robot behavior. For humanoid robots, control systems must handle complex dynamics, maintain balance, and coordinate multiple degrees of freedom. This section covers the essential control theory concepts for humanoid robotics applications.

## Introduction to Control Systems

A control system manages the behavior of a dynamic system by manipulating its inputs based on measurements of its outputs. In humanoid robotics, control systems regulate joint positions, maintain balance, coordinate movements, and achieve desired tasks.

### Basic Control System Components

A typical feedback control system consists of:

1. **Plant**: The system being controlled (e.g., a humanoid robot)
2. **Sensor**: Measures the system's output or state
3. **Controller**: Processes sensor data and computes control inputs
4. **Actuator**: Applies control inputs to the plant
5. **Reference input**: Desired behavior or trajectory
6. **Disturbance**: Unwanted external influences

### Open-Loop vs. Closed-Loop Control

**Open-Loop Control:**
```
Reference Input → Controller → Actuator → Plant → Output
```

In open-loop control, the system output does not affect the control action. This approach is simple but lacks the ability to correct for disturbances or modeling errors.

**Closed-Loop (Feedback) Control:**
```
Reference Input ──┐
                  ↓
              [Summing Junction] → Controller → Actuator → Plant → Output
                  ↑                                             ↓
                  └────────────── [Feedback Path] ← Sensor ←───┘
```

In closed-loop control, the system output is measured and compared to the reference, with the difference (error) used to adjust the control action.

## System Modeling

Mathematical models are essential for control system design and analysis.

### State-Space Representation

Many systems can be represented in state-space form:

```
ẋ(t) = f(x(t), u(t), t)
y(t) = h(x(t), u(t), t)
```

Where:
- x(t) is the state vector
- u(t) is the input vector
- y(t) is the output vector
- f and h are system functions

For linear time-invariant (LTI) systems:
```
ẋ = Ax + Bu
y = Cx + Du
```

Where A, B, C, and D are constant matrices.

#### Example: Simple Mass-Spring-Damper System

For a mass-spring-damper system:
```
m*ẍ + c*ẋ + k*x = F
```

Where m is mass, c is damping coefficient, k is spring constant, and F is applied force.

State-space representation with x₁ = x and x₂ = ẋ:
```
[x̂₁]   [ 0    1 ][x₁]   [0]
[x̂₂] = [-k/m -c/m][x₂] + [1/m] * F

y = [1 0][x₁]
           [x₂]
```

### Transfer Functions

For single-input, single-output (SISO) LTI systems, the transfer function is:
```
G(s) = Y(s)/U(s) = output Laplace transform / input Laplace transform
```

For the mass-spring-damper example:
```
G(s) = 1/(m*s² + c*s + k)
```

### Nonlinear Systems

Humanoid robots are inherently nonlinear systems due to:
- Complex kinematics (trigonometric functions)
- Nonlinear dynamics (Coriolis and centrifugal terms)
- Variable contact conditions (walking, manipulation)
- Actuator limitations

Linearization around operating points is often used for control design:
```
δẋ = A*δx + B*δu
δy = C*δx + D*δu
```

Where δ represents small deviations from the operating point.

## Stability Analysis

Stability is crucial for humanoid robot control systems.

### Lyapunov Stability

A system is stable if for every initial state close to the equilibrium, the system remains close to the equilibrium. It is asymptotically stable if it also converges to the equilibrium.

**Lyapunov's Direct Method:**
Find a function V(x) such that:
1. V(0) = 0 and V(x) > 0 for x ≠ 0 (positive definite)
2. V̇(x) ≤ 0 (non-increasing along system trajectories)

If such a function exists, the equilibrium is stable. If V̇(x) < 0 for x ≠ 0, it is asymptotically stable.

#### Example: Pendulum Stability

For an inverted pendulum system, a Lyapunov function might be:
```
V(x) = ½*m*l²*θ̇² + m*g*l*(1 - cos(θ))
```

This represents the total energy (kinetic + potential) of the system.

### Linear Stability

For linear systems ẋ = Ax, stability is determined by the eigenvalues of A:
- If all eigenvalues have negative real parts, the system is asymptotically stable
- If any eigenvalue has positive real part, the system is unstable
- If eigenvalues have zero real parts, further analysis is needed

### Routh-Hurwitz Criterion

For polynomial equations, the Routh-Hurwitz criterion provides a method to determine stability without computing roots.

For a characteristic polynomial:
```
aₙ*sⁿ + aₙ₋₁*sⁿ⁻¹ + ... + a₁*s + a₀ = 0
```

The system is stable if all coefficients are positive and all elements in the first column of the Routh array are positive.

## Feedback Control Techniques

### Proportional-Integral-Derivative (PID) Control

PID controllers are widely used in robotics due to their simplicity and effectiveness:

```
u(t) = Kp * e(t) + Ki * ∫e(t)dt + Kd * de(t)/dt
```

Where:
- e(t) is the error (reference - actual)
- Kp is the proportional gain
- Ki is the integral gain
- Kd is the derivative gain

In the Laplace domain:
```
Gc(s) = Kp + Ki/s + Kd*s = (Kd*s² + Kp*s + Ki)/s
```

#### PID Tuning Methods

**Ziegler-Nichols Method:**
1. Set Ki = Kd = 0
2. Increase Kp until the system oscillates with period Pu
3. Use tuning rules:
   - P: Kp = 0.5*Ku
   - PI: Kp = 0.45*Ku, Ki = 1.2*Kp/Pu
   - PID: Kp = 0.6*Ku, Ki = 2*Kp/Pu, Kd = Kp*Pu/8

Where Ku is the ultimate gain that causes sustained oscillations.

#### Applications in Humanoid Robotics

- **Joint position control**: Maintaining desired joint angles
- **Balance control**: Adjusting based on center of mass deviation
- **Trajectory following**: Tracking desired motion paths
- **Force control**: Regulating contact forces during manipulation

### State Feedback Control

For systems with full state measurement, state feedback can be used:
```
u = -Kx + r
```

Where K is the feedback gain matrix and r is the reference input.

**Pole Placement:**
Choose K such that the closed-loop system has desired eigenvalues (poles).

For ẋ = Ax + Bu, the closed-loop system is:
```
ẋ = (A - BK)x + Br
```

The characteristic equation is det(sI - A + BK) = 0. Choose K to place poles at desired locations.

**Linear Quadratic Regulator (LQR):**
Minimizes a quadratic cost function:
```
J = ∫[x^T Q x + u^T R u] dt
```

The optimal gain is K = R⁻¹B^T P, where P is the solution to the algebraic Riccati equation:
```
A^T P + PA - PBR⁻¹B^T P + Q = 0
```

### Observer Design

When not all states are measurable, observers estimate the full state:

**Luenberger Observer:**
```
x̂̇ = A*x̂ + B*u + L*(y - Ĉ*x̂)
```

Where x̂ is the estimated state and L is the observer gain matrix.

The observer error e = x - x̂ evolves as:
```
ė = (A - LC)e
```

Choose L such that A - LC is stable (e → 0).

## Advanced Control Techniques

### Model Predictive Control (MPC)

MPC solves an optimization problem at each time step to determine the optimal control input sequence:

```
min ∑[k=0 to N-1] (x_k^T Q x_k + u_k^T R u_k) + x_N^T P x_N
s.t. x_{k+1} = Ax_k + Bu_k
     x_min ≤ x_k ≤ x_max
     u_min ≤ u_k ≤ u_max
     y_min ≤ Cx_k ≤ y_max
```

Where N is the prediction horizon, and Q, R, P are weighting matrices.

#### MPC for Humanoid Robots

MPC is particularly useful for humanoid robots because it can handle:
- State and control constraints (joint limits, actuator limits)
- Prediction of future behavior (important for walking)
- Optimization of complex objectives (balance, energy, smoothness)
- Disturbance rejection

**Example: Walking Pattern Generation**
```
min ∑[k=0 to N-1] (||zmp_k - zmp_ref_k||² + ||com_k - com_ref_k||² + ||u_k||²)
s.t. com_{k+1} = A*com_k + B*zmp_k + C*u_k  (LIPM dynamics)
     zmp_k ∈ support_polygon_k              (stability constraint)
     ||u_k|| ≤ u_max                       (control effort limit)
```

### Adaptive Control

Adaptive control adjusts controller parameters in real-time based on system behavior:

**Model Reference Adaptive Control (MRAC):**
```
θ̇ = -Γ * φ(x) * e
u = -K(θ) * x
```

Where:
- θ are the adjustable parameters
- φ is the regressor vector
- Γ is the adaptation gain
- e is the tracking error

**Self-Tuning Regulator:**
Combines parameter estimation with optimal control:
1. Estimate system parameters online
2. Design controller based on estimated parameters
3. Apply control law

### Robust Control

Robust control handles model uncertainties and disturbances:

**H∞ Control:**
Minimizes the worst-case effect of disturbances:
```
||T_{wd}||_∞ = sup_ω σ̄(T_{wd}(jω))
```

Where T_{wd} is the transfer function from disturbances to errors, and σ̄ is the maximum singular value.

**μ-Synthesis:**
Handles structured uncertainties in the system model.

## Control Architectures for Humanoid Robots

### Hierarchical Control

Humanoid robots typically use multiple control layers:

```
High-Level Planner
       ↓
Trajectory Generator
       ↓
Low-Level Controller
       ↓
Actuators
```

**High-Level (1-10 Hz):**
- Task planning
- Path planning
- Gait pattern generation
- Whole-body motion planning

**Mid-Level (50-200 Hz):**
- Trajectory tracking
- Balance control
- Contact force regulation
- Gait adaptation

**Low-Level (100-1000 Hz):**
- Joint position/velocity control
- Torque control
- Hardware safety
- Fast disturbance rejection

### Whole-Body Control

Whole-body control considers the entire robot dynamics simultaneously:

```
min ||Ax - b||²
s.t. Cx = d
     x_min ≤ x ≤ x_max
```

Where x includes joint accelerations, contact forces, and other variables.

**Task Prioritization:**
```
min ||A₁x - b₁||²
s.t. A₂x = b₂
     Cx = d
```

Where A₁x = b₁ represents high-priority tasks and A₂x = b₂ represents low-priority tasks that must be satisfied exactly.

### Balance Control Strategies

**Zero Moment Point (ZMP) Control:**
```
ẍ_com = g/h * (x_com - x_zmp)
```

Where the ZMP is controlled to remain within the support polygon.

**Linear Inverted Pendulum Mode (LIPM):**
```
ẍ = ω² * (x - x_0)
```

Where ω² = g/h and x_0 is the virtual repellent point.

**Capture Point Control:**
```
Capture Point = CoM + √(h/g) * CoM_velocity
```

Control the next foot placement to the capture point.

## Implementation Considerations

### Discrete-Time Control

Digital controllers operate in discrete time:

```
x[k+1] = A_d * x[k] + B_d * u[k]
y[k] = C_d * x[k] + D_d * u[k]
```

Where A_d, B_d, C_d, D_d are discrete-time matrices related to continuous-time matrices by:
```
A_d = e^(AT_s)
B_d = ∫[0 to T_s] e^(Aτ) dτ * B
```

Where T_s is the sampling period.

### Sample Rate Selection

- **Too slow**: System may be unstable or poorly controlled
- **Too fast**: Computational burden, noise amplification
- **Rule of thumb**: Sample rate should be 10-20 times the system bandwidth

### Anti-Windup

Integrator windup occurs when the controller output saturates:

```
if |u_sat| < |u_unsat|:
    integral_error += e * T_s  # Only integrate when not saturated
```

### Feedforward Control

Combine feedback and feedforward:
```
u = u_feedback + u_feedforward
```

Feedforward terms can improve tracking performance:
```
u_ff = M(q)q̈_d + C(q,q̇)q̇_d + g(q)
```

Where q_d is the desired trajectory.

## Educational Applications

### Teaching Control Theory to Robotics Students

1. **Start with Intuition**: Use physical examples and analogies before mathematical formalism
2. **Progressive Complexity**: Begin with simple SISO systems before MIMO systems
3. **Simulation First**: Use simulation to demonstrate concepts before hardware implementation
4. **Hands-On Projects**: Implement controllers on real or simulated robots

### Common Student Difficulties

1. **Transfer Functions**: Understanding the relationship between time and frequency domains
2. **Stability Concepts**: Grasping the difference between BIBO and Lyapunov stability
3. **Controller Tuning**: Knowing how to adjust parameters for desired performance
4. **Discretization**: Understanding the effects of sampling in digital control

### Laboratory Exercises

1. **First-Order System Control**: Control a simple RC circuit or mass-damper system
2. **PID Tuning**: Practice tuning PID controllers for different systems
3. **Stability Analysis**: Analyze the stability of various systems
4. **Digital Control**: Implement controllers in discrete time
5. **Robot Control**: Apply control techniques to actual robotic systems

## Practical Implementation Tips

### Controller Design Process

1. **System Identification**: Understand the system dynamics and constraints
2. **Performance Requirements**: Define rise time, settling time, overshoot, steady-state error
3. **Controller Selection**: Choose appropriate control technique based on requirements
4. **Design**: Design controller parameters using analytical or optimization methods
5. **Simulation**: Test controller in simulation before hardware implementation
6. **Implementation**: Implement on hardware with appropriate safety measures
7. **Tuning**: Adjust parameters based on real-world performance
8. **Validation**: Verify performance meets requirements

### Safety Considerations

- **Saturation Limits**: Ensure control outputs don't exceed actuator capabilities
- **Stability Margins**: Design for robustness to model uncertainties
- **Emergency Stops**: Implement safety mechanisms to stop robot if control fails
- **Monitoring**: Continuously monitor system states for unexpected behavior

## Case Study: Balance Control for Humanoid Robots

Consider a humanoid robot that needs to maintain balance while standing:

**System Model (Simplified):**
```
m*ẍ_com = F_x
m*ÿ_com = F_y - mg
I*θ̈ = τ - m*g*l*sin(θ)
```

Where (x_com, y_com) is the center of mass, θ is the tilt angle, and (F_x, F_y, τ) are control forces/torques.

**Control Objective:**
Maintain θ ≈ 0, ẋ_com ≈ 0, ẏ_com ≈ 0

**Control Strategy:**
1. Use PID control for angle regulation
2. Use ZMP control for stability
3. Implement whole-body control for coordination

**Implementation:**
```
# Pseudocode for balance controller
while robot_is_active:
    # Measure current state
    current_state = get_robot_state()

    # Calculate desired ZMP based on CoM position/velocity
    desired_zmp = calculate_balance_reference(current_state)

    # Compute control torques using inverse dynamics
    control_torques = compute_balance_control(current_state, desired_zmp)

    # Apply torques to joints
    send_torques_to_actuators(control_torques)

    # Wait for next control cycle
    wait(control_period)
```

## Summary

This section covered the fundamental control theory concepts essential for humanoid robotics: system modeling in state-space and transfer function forms, stability analysis using Lyapunov methods and linear techniques, feedback control including PID and state feedback, and advanced techniques like MPC and adaptive control. The section also discussed control architectures specific to humanoid robots and practical implementation considerations. Understanding these concepts is crucial for designing effective control systems that enable humanoid robots to perform complex tasks while maintaining stability and safety.

## Exercises

[Exercises for this section are located in docs/theoretical-foundations/exercises.md]

## References

1. Ogata, K. (2021). *Modern Control Engineering* (6th ed.). Pearson. [Peer-reviewed]

2. Franklin, G. F., Powell, J. D., & Emami-Naeini, A. (2020). *Feedback Control of Dynamic Systems* (8th ed.). Pearson. [Peer-reviewed]

3. Slotine, J. J. E., & Li, W. (2020). *Applied Nonlinear Control*. Prentice Hall. [Peer-reviewed]

4. Siciliano, B., & Khatib, O. (Eds.). (2016). *Springer Handbook of Robotics* (2nd ed.). Springer. https://doi.org/10.1007/978-3-319-32552-1 [Peer-reviewed]

5. Grizzle, J. W., & Westervelt, E. R. (2022). Feedback control of dynamic bipedal robot locomotion. *CRC Press*. https://doi.org/10.1201/9781315220955 [Peer-reviewed]

6. Kajita, S., Kanehiro, F., Kaneko, K., Fujiwara, K., Harada, K., Yokoi, K., & Hirukawa, H. (2020). Biped walking pattern generation by using preview control of zero-moment point. *IEEE Transactions on Robotics*, 20(2), 182-195. https://doi.org/10.1109/ROBOT.2003.1241814 [Peer-reviewed]

7. Wieber, P. B., Tedrake, R., & Kuindersma, S. (2022). Modeling and control of legged robots. *Springer Handbook of Robotics*, 1203-1234. https://doi.org/10.1007/978-3-319-32552-1_50 [Peer-reviewed]