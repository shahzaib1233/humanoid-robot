---
title: Exercises - Advanced Control Systems
sidebar_label: Exercises
sidebar_position: 8
description: Exercises for the Advanced Control Systems chapter focusing on adaptive control, learning-based control, and multi-objective optimization
keywords: [exercises, advanced control, adaptive control, learning-based control, humanoid robotics]
---

# Exercises: Advanced Control Systems

These exercises are designed to reinforce the concepts covered in the Advanced Control Systems chapter. They range from theoretical problems to practical implementation challenges.

## Exercise 1: Adaptive Control for Uncertain Dynamics

### Problem Statement
Consider a humanoid robot leg with uncertain dynamics described by:
```
M(q)q̈ + C(q, q̇)q̇ + G(q) = τ + d(t)
```

Where `M(q)` is the uncertain inertia matrix, `C(q, q̇)` represents Coriolis and centrifugal terms, `G(q)` is gravity, `τ` is the control input, and `d(t)` is an unknown disturbance.

Design an adaptive controller that estimates the uncertain parameters and compensates for the disturbance.

### Tasks:
1. Formulate the parameter estimation problem using a linear-in-parameters model
2. Derive the adaptation laws using Lyapunov stability theory
3. Implement the controller in simulation
4. Test the controller's performance under varying payloads

### Solution Approach:
```python
import numpy as np

class AdaptiveLegController:
    def __init__(self, num_joints=6):
        self.n = num_joints
        self.theta = np.random.normal(0, 0.1, 20)  # Adaptive parameters
        self.P = np.eye(20) * 1000  # Covariance matrix
        self.gamma = 0.1  # Adaptation gain

    def regressor(self, q, q_dot, q_ddot_desired):
        """Compute the regressor matrix Y such that Y*theta = dynamics"""
        # Simplified regressor - in practice, this would be more complex
        phi = np.concatenate([
            q,
            q_dot,
            q_ddot_desired,
            np.sin(q),
            np.cos(q),
            q * q_dot,
            np.sin(q_dot)
        ])
        return phi

    def control_law(self, q, q_dot, q_desired, q_dot_desired, q_ddot_desired):
        """Compute control input using adaptive control law"""
        # Tracking error
        q_tilde = q - q_desired
        q_dot_tilde = q_dot - q_dot_desired

        # Adaptive parameter estimation
        phi = self.regressor(q, q_dot, q_ddot_desired)
        Y = np.column_stack([phi, np.eye(len(phi))])  # Simplified

        # Control law: tau = Y*theta_hat + K*v
        K = np.eye(self.n) * 10  # Feedback gain
        v = q_ddot_desired - K @ q_tilde - K @ q_dot_tilde  # Auxiliary variable

        # Estimate dynamics
        tau = Y @ self.theta + K @ (q_dot - q_dot_desired)

        # Update parameters using least squares
        L = self.P @ phi / (1 + phi.T @ self.P @ phi)
        error = Y @ self.theta - (q_ddot_desired + K @ q_tilde + K @ q_dot_tilde)
        self.theta = self.theta - L * error
        self.P = self.P - np.outer(L, phi) @ self.P

        return tau
```

### Expected Outcomes:
- The controller should maintain tracking performance despite parameter uncertainties
- Parameter estimates should converge to reasonable values
- Robustness to disturbances should be demonstrated

## Exercise 2: Reinforcement Learning for Walking Control

### Problem Statement
Implement a reinforcement learning algorithm to learn stable walking patterns for a simplified bipedal model.

### Tasks:
1. Define the state space (joint angles, velocities, IMU readings)
2. Define the action space (torque commands or desired joint positions)
3. Design a reward function that promotes stable, efficient walking
4. Implement a policy gradient algorithm to learn the walking policy
5. Test the learned policy in simulation

### Solution Approach:
```python
import numpy as np
import torch
import torch.nn as nn

class WalkingPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, state):
        return self.network(state)

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4):
        self.actor = WalkingPolicy(state_dim, action_dim)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_mean = self.actor(state_tensor)
        action_distribution = torch.distributions.Normal(action_mean, 0.5)
        action = action_distribution.sample()
        log_prob = action_distribution.log_prob(action).sum()
        return action.detach().numpy(), log_prob.detach().numpy()

    def update(self, states, actions, rewards, log_probs, next_states):
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        log_probs = torch.FloatTensor(log_probs)
        next_states = torch.FloatTensor(next_states)

        # Compute advantages
        with torch.no_grad():
            values = self.critic(states).squeeze()
            next_values = self.critic(next_states).squeeze()
            advantages = rewards + 0.99 * next_values - values

        # Update actor
        curr_log_probs = self.actor(states)
        ratio = torch.exp(curr_log_probs - log_probs)
        actor_loss = -torch.min(
            ratio * advantages,
            torch.clamp(ratio, 0.8, 1.2) * advantages
        ).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update critic
        critic_loss = advantages.pow(2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
```

### Expected Outcomes:
- The robot should learn to walk stably with minimal falls
- Walking efficiency (energy consumption) should improve over training
- The learned policy should generalize to slight variations in terrain

## Exercise 3: Multi-Objective Control for Balance and Manipulation

### Problem Statement
Design a controller that simultaneously maintains balance while performing a manipulation task. The controller must balance competing objectives: stability, manipulation accuracy, and energy efficiency.

### Tasks:
1. Formulate the multi-objective optimization problem
2. Implement a weighted sum approach to combine objectives
3. Design a Model Predictive Control (MPC) framework for this problem
4. Test the controller with different weight combinations

### Solution Approach:
```python
import numpy as np
from scipy.optimize import minimize

class MultiTaskController:
    def __init__(self, horizon=20, dt=0.02):
        self.horizon = horizon
        self.dt = dt
        self.n_states = 12  # Simplified state vector
        self.n_controls = 6  # Control inputs

    def predict_states(self, initial_state, control_sequence):
        """Predict state trajectory given initial state and control sequence"""
        states = [initial_state]
        current_state = initial_state.copy()

        for control in control_sequence:
            # Simplified dynamics model
            next_state = self.dynamics_step(current_state, control)
            states.append(next_state)
            current_state = next_state

        return states

    def dynamics_step(self, state, control):
        """Simplified dynamics for prediction"""
        # In practice, this would use a more accurate model
        next_state = state + control * self.dt
        return next_state

    def stability_cost(self, state_trajectory):
        """Compute cost related to balance stability"""
        cost = 0
        for state in state_trajectory:
            # Simplified: penalize CoM deviation from nominal position
            com_deviation = np.linalg.norm(state[0:3])  # x, y, z position
            cost += com_deviation**2
        return cost

    def manipulation_cost(self, state_trajectory, target_pose):
        """Compute cost related to manipulation task"""
        cost = 0
        for state in state_trajectory:
            # Simplified: penalize end-effector deviation from target
            ee_pos = state[6:9]  # Simplified end-effector position
            deviation = np.linalg.norm(ee_pos - target_pose[0:3])
            cost += deviation**2
        return cost

    def efficiency_cost(self, control_trajectory):
        """Compute cost related to energy efficiency"""
        cost = 0
        for control in control_trajectory:
            cost += np.sum(control**2)  # Quadratic control cost
        return cost

    def optimize_multi_task(self, initial_state, target_pose, weights):
        """
        Optimize control sequence for multiple tasks
        weights = {'stability': w1, 'manipulation': w2, 'efficiency': w3}
        """
        def objective(control_flat):
            # Reshape flat control vector
            controls = control_flat.reshape((self.horizon, self.n_controls))

            # Predict state trajectory
            states = self.predict_states(initial_state, controls)

            # Compute costs
            stability = self.stability_cost(states)
            manipulation = self.manipulation_cost(states, target_pose)
            efficiency = self.efficiency_cost(controls)

            # Weighted sum
            total_cost = (weights['stability'] * stability +
                         weights['manipulation'] * manipulation +
                         weights['efficiency'] * efficiency)

            return total_cost

        # Initial guess for controls
        initial_controls = np.zeros(self.horizon * self.n_controls)

        # Optimize
        result = minimize(objective, initial_controls, method='SLSQP')

        if result.success:
            optimal_controls = result.x.reshape((self.horizon, self.n_controls))
            return optimal_controls[0]  # Return first control
        else:
            return np.zeros(self.n_controls)  # Return zero if failed
```

### Expected Outcomes:
- The robot should maintain balance while performing manipulation
- Different weight combinations should produce different behaviors
- The controller should handle trade-offs between objectives gracefully

## Exercise 4: Robust Control Design

### Problem Statement
Design a robust controller for a humanoid robot that maintains performance despite model uncertainties and external disturbances.

### Tasks:
1. Model the uncertainties in the robot dynamics
2. Design an H-infinity controller to handle the uncertainties
3. Analyze the robustness margins of the controller
4. Test the controller under various disturbance conditions

### Solution Approach:
```python
class RobustController:
    def __init__(self, nominal_model, uncertainty_bounds):
        self.nominal_A = nominal_model['A']
        self.nominal_B = nominal_model['B']
        self.uncertainty_bounds = uncertainty_bounds
        self.gamma = 1.0  # Performance bound

        # Design robust controller using H-infinity synthesis
        # (In practice, would use control toolbox like python-control)
        self.K = self.design_hinf_controller()

    def design_hinf_controller(self):
        """Design H-infinity controller (simplified approach)"""
        # This would involve solving Riccati equations in practice
        # For this exercise, we'll use a simplified approach
        A, B = self.nominal_A, self.nominal_B

        # Design state feedback gain using LQR as a starting point
        Q = np.eye(A.shape[0]) * 10  # State weighting
        R = np.eye(B.shape[1]) * 0.1  # Control weighting

        # Solve algebraic Riccati equation (simplified)
        P = np.eye(A.shape[0]) * 5  # Solution to ARE
        K = np.linalg.inv(R) @ B.T @ P

        return K

    def robustness_analysis(self):
        """Analyze robustness properties"""
        # Compute robustness margins
        # This would involve mu-analysis or similar techniques
        stability_margin = 0.8  # Simplified result
        performance_margin = 0.9

        return {
            'stability_margin': stability_margin,
            'performance_margin': performance_margin,
            'robustness_level': 'High' if stability_margin > 0.7 else 'Low'
        }

    def control_step(self, state, disturbance_estimate=0):
        """Compute robust control input"""
        # State feedback with robust design
        control = -self.K @ state

        # Add disturbance rejection term if available
        if disturbance_estimate is not None:
            control += disturbance_estimate  # Simplified disturbance compensation

        return control
```

### Expected Outcomes:
- The controller should maintain stability under model uncertainties
- Performance should degrade gracefully with increasing disturbances
- Robustness margins should be quantifiable and acceptable

## Exercise 5: Implementation Challenge - Real-Time Constraints

### Problem Statement
Implement a real-time control system for humanoid robot balance that meets strict timing requirements (1000 Hz control frequency).

### Tasks:
1. Design a control loop that meets real-time requirements
2. Implement timing monitoring and violation detection
3. Add fail-safe mechanisms for timing violations
4. Test the system under computational load

### Solution Approach:
```python
import time
import threading
import numpy as np

class RealTimeBalanceController:
    def __init__(self, frequency=1000):
        self.frequency = frequency
        self.period = 1.0 / frequency
        self.last_time = time.time()
        self.timing_violations = 0
        self.max_execution_time = 0.0005  # 0.5ms max execution time
        self.control_thread = None
        self.running = False

    def control_loop(self):
        """Main control loop running at specified frequency"""
        while self.running:
            start_time = time.time()

            # Perform control computation
            sensor_data = self.read_sensors()
            control_output = self.balance_control(sensor_data)
            self.send_actuator_commands(control_output)

            # Calculate execution time
            execution_time = time.time() - start_time

            if execution_time > self.max_execution_time:
                self.timing_violations += 1
                print(f"Timing violation: execution took {execution_time:.6f}s")

            # Wait for next cycle
            sleep_time = self.period - execution_time
            if sleep_time > 0:
                time.sleep(sleep_time)

            # Monitor timing
            actual_period = time.time() - start_time
            if actual_period > self.period * 1.1:  # 10% tolerance
                self.timing_violations += 1

    def balance_control(self, sensor_data):
        """Balance control algorithm"""
        # Simplified balance control
        com_error = sensor_data.get('com_error', 0)
        angular_error = sensor_data.get('angular_error', 0)

        # Simple PD control
        kp_com, kd_com = 100, 10
        kp_ang, kd_ang = 50, 5

        control_output = (
            kp_com * com_error + kd_com * sensor_data.get('com_velocity', 0) +
            kp_ang * angular_error + kd_ang * sensor_data.get('angular_velocity', 0)
        )

        return control_output

    def start_control(self):
        """Start the real-time control loop"""
        self.running = True
        self.control_thread = threading.Thread(target=self.control_loop)
        self.control_thread.start()

    def stop_control(self):
        """Stop the real-time control loop"""
        self.running = False
        if self.control_thread:
            self.control_thread.join()

    def get_performance_stats(self):
        """Get real-time performance statistics"""
        return {
            'timing_violations': self.timing_violations,
            'target_frequency': self.frequency,
            'target_period': self.period,
            'max_execution_time': self.max_execution_time
        }

    def read_sensors(self):
        """Simulate sensor reading"""
        return {
            'com_error': np.random.normal(0, 0.01),
            'angular_error': np.random.normal(0, 0.005),
            'com_velocity': np.random.normal(0, 0.001),
            'angular_velocity': np.random.normal(0, 0.001)
        }

    def send_actuator_commands(self, commands):
        """Simulate sending commands to actuators"""
        pass  # In real implementation, would send to hardware
```

### Expected Outcomes:
- The control loop should run at or near the target frequency
- Timing violations should be minimal (less than 1% of cycles)
- The system should maintain balance performance under timing constraints

## Solutions and Discussion

### Exercise 1 Discussion:
Adaptive control is crucial for humanoid robots because their dynamics change due to varying payloads, wear, and environmental conditions. The key challenge is ensuring parameter convergence while maintaining stability. The regressor formulation is critical - it must be persistently exciting for parameter convergence to occur.

### Exercise 2 Discussion:
Reinforcement learning offers the potential for humanoid robots to learn complex behaviors that are difficult to program explicitly. However, safety during learning is a major concern. Techniques like safe exploration, simulation-to-reality transfer, and policy verification are important considerations.

### Exercise 3 Discussion:
Multi-objective control reflects the reality that humanoid robots must balance multiple competing requirements. The weighting approach is simple but may not capture complex preference structures. Advanced techniques like Pareto optimization or hierarchical task frameworks may be more appropriate for complex scenarios.

### Exercise 4 Discussion:
Robust control is essential for safety-critical humanoid applications. The trade-off between robustness and performance is fundamental - more robust controllers tend to be more conservative. H-infinity methods provide guaranteed performance bounds but may be conservative in practice.

### Exercise 5 Discussion:
Real-time implementation is critical for humanoid robot control. The 1000 Hz requirement for joint control is typical, while higher-level tasks may run at lower frequencies. Proper real-time operating systems, deterministic algorithms, and careful system design are essential.

## References

1. Slotine, J. J. E., & Li, W. (2020). *Applied Nonlinear Control*. Prentice Hall. [Peer-reviewed]

2. Sutton, R. S., & Barto, A. G. (2022). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. [Peer-reviewed]

3. Zhou, K., & Doyle, J. C. (2020). *Essentials of Robust Control*. Prentice Hall. [Peer-reviewed]

4. Rawlings, J. B., & Mayne, D. Q. (2021). *Model Predictive Control: Theory, Computation, and Design*. Nob Hill Publishing. [Peer-reviewed]

5. Siciliano, B., & Khatib, O. (2016). *Springer Handbook of Robotics*. Springer. [Peer-reviewed]

## Summary

These exercises covered key aspects of advanced control for humanoid robots:
- Adaptive control for handling uncertainties
- Learning-based methods for complex behaviors
- Multi-objective optimization for competing requirements
- Robust control for safety and reliability
- Real-time implementation considerations

Each exercise builds on the theoretical concepts while addressing practical implementation challenges specific to humanoid robotics.