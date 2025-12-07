---
title: Exercises - Machine Learning for Robotics
sidebar_label: Exercises
sidebar_position: 10
description: Exercises for the Machine Learning for Robotics chapter focusing on perception, control, and decision making
keywords: [exercises, machine learning, robotics, perception, control, neural networks]
---

# Exercises: Machine Learning for Robotics

These exercises are designed to reinforce the concepts covered in the Machine Learning for Robotics chapter. They range from theoretical problems to practical implementation challenges.

## Exercise 1: CNN for Robotic Vision

### Problem Statement
Implement a convolutional neural network to classify objects in a robot's camera feed. The network should be able to recognize 10 different household objects with at least 80% accuracy.

### Tasks:
1. Design a CNN architecture suitable for robotic vision
2. Implement data augmentation techniques for robustness
3. Train the network on a dataset of robot-captured images
4. Evaluate the network's performance and robustness to lighting changes

### Solution Approach:
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np

class RobotVisionCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # Feature extraction layers
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Second block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Third block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Fourth block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))  # Ensure consistent output size
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

class RoboticVisionDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def train_vision_network():
    # Define data transforms for training (with augmentation)
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Define data transforms for validation (without augmentation)
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Initialize model
    model = RobotVisionCNN(num_classes=10)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Training loop (simplified)
    model.train()
    for epoch in range(50):  # 50 epochs
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        avg_loss = running_loss / len(train_loader)

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')

    return model

# Example usage
# model = train_vision_network()
```

### Expected Outcomes:
- Network should achieve at least 80% accuracy on validation set
- Robust to lighting variations and object orientations
- Efficient enough for real-time inference on robotic hardware

## Exercise 2: Reinforcement Learning for Robot Navigation

### Problem Statement
Implement a reinforcement learning agent to navigate a robot through a 2D environment with obstacles to reach a target location.

### Tasks:
1. Define the state space (robot position, sensor readings, target position)
2. Define the action space (movement commands)
3. Design a reward function that encourages efficient navigation
4. Implement a Deep Q-Network (DQN) algorithm
5. Test the agent in various environments

### Solution Approach:
```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class NavigationDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        return self.network(state)

class NavigationEnvironment:
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self.robot_pos = None
        self.target_pos = None
        self.obstacles = []
        self.max_steps = 100

    def reset(self):
        # Randomly place robot and target
        self.robot_pos = np.random.randint(0, self.width, 2)
        self.target_pos = np.random.randint(0, self.width, 2)

        # Ensure robot and target are not at the same position
        while np.array_equal(self.robot_pos, self.target_pos):
            self.target_pos = np.random.randint(0, self.width, 2)

        # Generate random obstacles
        self.obstacles = []
        num_obstacles = np.random.randint(5, 15)
        for _ in range(num_obstacles):
            obs = np.random.randint(0, self.width, 2)
            # Ensure obstacles are not on robot or target
            while (np.array_equal(obs, self.robot_pos) or
                   np.array_equal(obs, self.target_pos) or
                   any(np.array_equal(obs, o) for o in self.obstacles)):
                obs = np.random.randint(0, self.width, 2)
            self.obstacles.append(obs)

        self.step_count = 0
        return self.get_state()

    def get_state(self):
        # State: [robot_x, robot_y, target_x, target_y, distance_to_target]
        dist_to_target = np.linalg.norm(self.robot_pos - self.target_pos)
        state = np.concatenate([
            self.robot_pos.astype(float),
            self.target_pos.astype(float),
            [dist_to_target]
        ])
        return state

    def step(self, action):
        # Actions: 0=up, 1=right, 2=down, 3=left
        new_pos = self.robot_pos.copy()

        if action == 0:  # Up
            new_pos[1] = max(0, new_pos[1] - 1)
        elif action == 1:  # Right
            new_pos[0] = min(self.width - 1, new_pos[0] + 1)
        elif action == 2:  # Down
            new_pos[1] = min(self.height - 1, new_pos[1] + 1)
        elif action == 3:  # Left
            new_pos[0] = max(0, new_pos[0] - 1)

        # Check if new position is obstacle
        if any(np.array_equal(new_pos, obs) for obs in self.obstacles):
            # Stay in current position if obstacle
            reward = -10  # Penalty for hitting obstacle
        else:
            self.robot_pos = new_pos
            reward = -0.1  # Small time penalty

            # Calculate distance to target
            dist_to_target = np.linalg.norm(self.robot_pos - self.target_pos)

            # Reward based on getting closer to target
            reward += -dist_to_target * 0.01

            # Check if reached target
            if np.array_equal(self.robot_pos, self.target_pos):
                reward = 100  # Large reward for reaching target
                done = True
            else:
                done = False

        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True
            reward -= 50  # Penalty for not reaching target in time

        return self.get_state(), reward, done

class NavigationDQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Neural networks
        self.q_network = NavigationDQN(state_dim, action_dim)
        self.target_network = NavigationDQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Experience replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 32

        # Update target network
        self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_dim)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_navigation_agent():
    env = NavigationEnvironment()
    agent = NavigationDQNAgent(state_dim=5, action_dim=4)  # 5D state, 4 actions

    episodes = 1000

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0

        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            agent.replay()

            state = next_state
            total_reward += reward
            steps += 1

            if done:
                break

        # Update target network periodically
        if episode % 10 == 0:
            agent.update_target_network()

        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Steps: {steps}, Epsilon: {agent.epsilon:.3f}")

# Uncomment to run training
# train_navigation_agent()
```

### Expected Outcomes:
- Agent should learn to navigate to target efficiently
- Should avoid obstacles and find optimal paths
- Performance should improve over training episodes

## Exercise 3: Imitation Learning for Robotic Manipulation

### Problem Statement
Implement an imitation learning system where a robot learns to perform a manipulation task by observing expert demonstrations.

### Tasks:
1. Collect expert demonstrations of a manipulation task
2. Implement a behavioral cloning algorithm
3. Evaluate the learned policy
4. Implement DAgger (Dataset Aggregation) for improved performance

### Solution Approach:
```python
class ManipulationImitationLearner:
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Policy network for behavioral cloning
        self.policy_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()

        # Dataset for demonstrations
        self.demonstration_buffer = []
        self.training_buffer = []

    def add_demonstration(self, states, actions):
        """Add a demonstration trajectory to the buffer"""
        for state, action in zip(states, actions):
            self.demonstration_buffer.append((state, action))

    def behavioral_cloning_update(self):
        """Update policy using behavioral cloning"""
        if len(self.training_buffer) < 32:
            return

        # Sample batch from training buffer
        batch = random.sample(self.training_buffer, min(32, len(self.training_buffer)))
        states = torch.FloatTensor([item[0] for item in batch])
        actions = torch.FloatTensor([item[1] for item in batch])

        # Update policy
        self.optimizer.zero_grad()
        predicted_actions = self.policy_network(states)
        loss = self.criterion(predicted_actions, actions)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def dagger_update(self, env, expert_policy, num_rollouts=5):
        """Perform DAgger update by collecting new data from current policy"""
        for _ in range(num_rollouts):
            state = env.reset()
            states = []
            actions = []

            done = False
            while not done:
                # Get action from current policy
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    policy_action = self.policy_network(state_tensor).numpy().squeeze()

                # Get expert action for the same state
                expert_action = expert_policy(state)

                states.append(state)
                actions.append(expert_action)  # Use expert action

                # Take step with policy action (to collect diverse states)
                state, _, done = env.step(policy_action)

            # Add to training buffer
            for s, a in zip(states, actions):
                self.training_buffer.append((s, a))

        # Update policy with new data
        return self.behavioral_cloning_update()

def create_expert_policy():
    """Create a simple expert policy for demonstration"""
    def expert_policy(state):
        # Simplified expert: move towards target
        robot_pos = state[:2]  # Robot position
        target_pos = state[2:4]  # Target position

        # Calculate direction to target
        direction = target_pos - robot_pos
        distance = np.linalg.norm(direction)

        if distance < 0.1:  # Close enough
            return np.array([0.0, 0.0])  # Stop

        # Normalize direction and scale
        direction = direction / distance
        action = direction * 0.1  # Small step

        return action

    return expert_policy

def train_imitation_learner():
    """Train the imitation learner"""
    learner = ManipulationImitationLearner(state_dim=4, action_dim=2)
    expert_policy = create_expert_policy()

    # Create simple environment for demonstration
    env = NavigationEnvironment(width=5, height=5)

    # Collect initial demonstrations
    for _ in range(10):  # 10 demonstration trajectories
        state = env.reset()
        states = [state]
        actions = []

        done = False
        while not done:
            action = expert_policy(state)
            actions.append(action)
            state, _, done = env.step(action)
            if not done:  # Don't add terminal state as next state
                states.append(state)

        learner.add_demonstration(states, actions)

    # Train using behavioral cloning
    for epoch in range(100):
        loss = learner.behavioral_cloning_update()
        if epoch % 20 == 0 and loss:
            print(f"BC Epoch {epoch}, Loss: {loss:.4f}")

    # Optionally, perform DAgger updates
    for iteration in range(5):
        dagger_loss = learner.dagger_update(env, expert_policy)
        if dagger_loss:
            print(f"DAgger Iteration {iteration}, Loss: {dagger_loss:.4f}")

# Uncomment to run training
# train_imitation_learner()
```

### Expected Outcomes:
- Behavioral cloning should learn basic task execution
- DAgger should improve performance by collecting diverse training data
- Policy should generalize to new starting positions

## Exercise 4: Sensor Fusion with Neural Networks

### Problem Statement
Implement a neural network that fuses data from multiple sensors (camera, LIDAR, IMU) to improve robot state estimation.

### Tasks:
1. Design a network architecture for sensor fusion
2. Implement attention mechanisms to weight sensor contributions
3. Train the network to estimate robot pose
4. Evaluate the fused estimate vs. individual sensor estimates

### Solution Approach:
```python
class SensorFusionNetwork(nn.Module):
    def __init__(self, camera_dim=512, lidar_dim=256, imu_dim=6, pose_dim=6, hidden_dim=256):
        super().__init__()

        # Process each sensor modality separately
        self.camera_processor = nn.Sequential(
            nn.Linear(camera_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU()
        )

        self.lidar_processor = nn.Sequential(
            nn.Linear(lidar_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU()
        )

        self.imu_processor = nn.Sequential(
            nn.Linear(imu_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Attention mechanism to weight sensor contributions
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim//2, num_heads=4)

        # Sensor fusion layer
        self.fusion = nn.Sequential(
            nn.Linear((hidden_dim//2)*2 + 32, hidden_dim),  # Combined sensor dims
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, pose_dim)
        )

        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim//2, 64),
            nn.ReLU(),
            nn.Linear(64, pose_dim),
            nn.Softplus()  # Ensure positive uncertainty
        )

    def forward(self, camera_features, lidar_features, imu_features):
        # Process each sensor modality
        cam_emb = self.camera_processor(camera_features)
        lidar_emb = self.lidar_processor(lidar_features)
        imu_emb = self.imu_processor(imu_features)

        # Stack embeddings for attention
        sensor_embeddings = torch.stack([cam_emb, lidar_emb], dim=0)  # Only modalities with attention

        # Apply attention mechanism
        attended_features, attention_weights = self.attention(
            sensor_embeddings, sensor_embeddings, sensor_embeddings
        )

        # Flatten attended features
        attended_flat = attended_features.permute(1, 0, 2).contiguous().view(
            attended_features.size(1), -1
        )

        # Concatenate with IMU features and apply fusion
        combined_features = torch.cat([attended_flat, imu_emb], dim=1)
        pose_estimate = self.fusion(combined_features)
        uncertainty = self.uncertainty_head(self.fusion[2:4](combined_features))  # Use intermediate features

        return {
            'pose': pose_estimate,
            'uncertainty': uncertainty,
            'attention_weights': attention_weights
        }

def train_sensor_fusion():
    """Train the sensor fusion network"""
    model = SensorFusionNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    # Simulated training data
    batch_size = 32
    num_batches = 100

    for epoch in range(50):
        total_loss = 0

        for batch in range(num_batches):
            # Generate simulated sensor data
            camera_data = torch.randn(batch_size, 512)
            lidar_data = torch.randn(batch_size, 256)
            imu_data = torch.randn(batch_size, 6)
            true_poses = torch.randn(batch_size, 6)  # [x, y, z, roll, pitch, yaw]

            optimizer.zero_grad()

            # Forward pass
            output = model(camera_data, lidar_data, imu_data)
            pose_pred = output['pose']

            # Compute loss
            loss = criterion(pose_pred, true_poses)

            # Add uncertainty-based loss (optional)
            uncertainty = output['uncertainty']
            uncertainty_loss = torch.mean(uncertainty)  # Regularize uncertainty
            total_loss_batch = loss + 0.01 * uncertainty_loss

            # Backward pass
            total_loss_batch.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")

# Example usage
# train_sensor_fusion()
```

### Expected Outcomes:
- Fused estimate should be more accurate than individual sensors
- Attention mechanism should adaptively weight more reliable sensors
- Uncertainty estimates should correlate with actual estimation errors

## Exercise 5: Safe Reinforcement Learning

### Problem Statement
Implement a reinforcement learning algorithm with safety constraints for robotic control, ensuring the robot avoids dangerous states during learning.

### Tasks:
1. Define safety constraints for the robotic system
2. Implement constrained policy optimization
3. Design a safety critic to evaluate constraint violations
4. Test the algorithm in a simulated environment

### Solution Approach:
```python
class SafeRLAgent:
    def __init__(self, state_dim, action_dim, safety_constraints, hidden_dim=256):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.safety_constraints = safety_constraints

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Bound actions
        )

        # Critic network for value estimation
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Cost critic for safety constraint estimation
        self.cost_critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.cost_optimizer = optim.Adam(self.cost_critic.parameters(), lr=1e-3)

        # Replay buffer
        self.buffer = deque(maxlen=10000)

        # Safety parameters
        self.safety_budget = 0.1  # Maximum allowed constraint violation rate
        self.cost_limit = 0.05    # Maximum expected cost per step

    def act(self, state, evaluate_safety=True):
        """Select action with safety considerations"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            action_mean = self.actor(state_tensor)

        # If evaluating safety, check if action leads to safe state
        if evaluate_safety:
            # Predict next state with action
            predicted_next_state = self.predict_next_state(state, action_mean.numpy().squeeze())

            # Check safety constraints
            if self.is_state_safe(predicted_next_state):
                return action_mean.numpy().squeeze()
            else:
                # Return safe fallback action
                return self.get_safe_action(state)
        else:
            return action_mean.numpy().squeeze()

    def predict_next_state(self, state, action):
        """Predict next state given current state and action (simplified model)"""
        # This would use a learned dynamics model in practice
        # For this exercise, using a simple integration
        next_state = state.copy()
        next_state[:2] += action[:2] * 0.1  # Update position based on action
        return next_state

    def is_state_safe(self, state):
        """Check if state satisfies safety constraints"""
        return self.safety_constraints.is_safe(state)

    def get_safe_action(self, state):
        """Get a safe fallback action"""
        # Return action that moves away from unsafe regions
        # This is a simplified implementation
        return np.zeros(self.action_dim)

    def update(self, batch):
        """Update the agent with a batch of experiences"""
        states = torch.FloatTensor([b[0] for b in batch])
        actions = torch.FloatTensor([b[1] for b in batch])
        rewards = torch.FloatTensor([b[2] for b in batch]).unsqueeze(1)
        next_states = torch.FloatTensor([b[3] for b in batch])
        dones = torch.BoolTensor([b[4] for b in batch]).unsqueeze(1)
        costs = torch.FloatTensor([b[5] for b in batch]).unsqueeze(1)  # Safety costs

        # Update critic
        with torch.no_grad():
            next_values = self.critic(next_states)
            target_values = rewards + 0.99 * next_values * (~dones)

        current_values = self.critic(states)
        critic_loss = nn.MSELoss()(current_values, target_values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update cost critic
        with torch.no_grad():
            next_cost_values = self.cost_critic(next_states)
            target_cost_values = costs + 0.99 * next_cost_values * (~dones)

        current_cost_values = self.cost_critic(states)
        cost_critic_loss = nn.MSELoss()(current_cost_values, target_cost_values)

        self.cost_optimizer.zero_grad()
        cost_critic_loss.backward()
        self.cost_optimizer.step()

        # Update actor (simplified - in practice, this would use more sophisticated constrained optimization)
        values = self.critic(states)
        advantages = rewards - values

        # Consider safety costs in advantages
        cost_advantages = costs - self.cost_critic(states)

        # Actor loss: maximize reward while minimizing safety cost
        actor_loss = -(advantages - 0.1 * cost_advantages).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

class SafetyConstraints:
    """Define safety constraints for the robotic system"""
    def __init__(self):
        # Define safe regions (example: avoid certain positions)
        self.safe_x_range = (-5.0, 5.0)
        self.safe_y_range = (-5.0, 5.0)
        self.max_velocity = 2.0

    def is_safe(self, state):
        """Check if state is safe"""
        x, y = state[0], state[1]
        vx, vy = state[2], state[3] if len(state) > 3 else 0, 0

        # Check position constraints
        pos_safe = (self.safe_x_range[0] <= x <= self.safe_x_range[1] and
                   self.safe_y_range[0] <= y <= self.safe_y_range[1])

        # Check velocity constraints
        vel_safe = (abs(vx) <= self.max_velocity and abs(vy) <= self.max_velocity)

        return pos_safe and vel_safe

def train_safe_agent():
    """Train the safe RL agent"""
    safety_constraints = SafetyConstraints()
    agent = SafeRLAgent(state_dim=4, action_dim=2, safety_constraints=safety_constraints)

    # Simulated training loop
    for episode in range(1000):
        # This would involve running the agent in an environment
        # Collecting experiences with safety costs
        # And calling agent.update() with the experiences
        pass

# Example usage
# train_safe_agent()
```

### Expected Outcomes:
- Agent should learn to achieve tasks while respecting safety constraints
- Constraint violation rate should remain below safety budget
- Performance should be competitive with unsafe methods while being safer

## Solutions and Discussion

### Exercise 1 Discussion:
CNNs for robotic vision require careful design to balance accuracy and computational efficiency. Data augmentation is crucial for robustness to lighting and viewpoint variations. The architecture should consider the computational constraints of robotic platforms while maintaining sufficient representational power.

### Exercise 2 Discussion:
Reinforcement learning for navigation requires careful reward design to avoid local optima. Experience replay helps with sample efficiency, and target networks improve training stability. The exploration-exploitation trade-off is critical for learning effective navigation policies.

### Exercise 3 Discussion:
Imitation learning can bootstrap robot learning from expert demonstrations but may suffer from compounding errors. DAgger addresses this by iteratively collecting data from the current policy and expert corrections. The quality of demonstrations significantly impacts learning performance.

### Exercise 4 Discussion:
Sensor fusion networks must effectively combine information from heterogeneous sensors. Attention mechanisms can adaptively weight sensor contributions based on reliability. Uncertainty estimation is crucial for safe robot operation when sensor fusion fails.

### Exercise 5 Discussion:
Safe RL balances task performance with constraint satisfaction. Constrained optimization methods ensure safety during learning. The design of safety constraints and cost functions is critical for effective safe learning.

## References

1. Kober, J., Bagnell, J. A., & Peters, J. (2013). Reinforcement learning in robotics: A survey. *The International Journal of Robotics Research*, 32(11), 1238-1274. https://doi.org/10.1177/0278364913495721 [Peer-reviewed]

2. Levine, S., Pastor, P., Krizhevsky, A., & Quillen, D. (2016). Learning hand-eye coordination for robotic grasping with deep learning and large-scale data collection. *The International Journal of Robotics Research*, 37(4-5), 421-436. https://doi.org/10.1177/0278364918774458 [Peer-reviewed]

3. Rajeswaran, A., Kumar, V., Gupta, A., & Todorov, E. (2017). Learning complex dexterous manipulation with deep reinforcement learning and demonstrations. *Proceedings of the 1st Annual Conference on Robot Learning*, 170-183. [Peer-reviewed]

4. Achiam, J., et al. (2017). Constrained policy optimization. *International Conference on Machine Learning*, 22-31. [Peer-reviewed]

5. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. [Peer-reviewed]

## Summary

These exercises covered key aspects of machine learning in robotics:
- Vision systems using CNNs
- Navigation using reinforcement learning
- Manipulation using imitation learning
- Sensor fusion with attention mechanisms
- Safe learning with constraint satisfaction

Each exercise builds on theoretical concepts while addressing practical implementation challenges specific to robotic applications. The solutions demonstrate how ML techniques can be adapted for the unique requirements of robotic systems.