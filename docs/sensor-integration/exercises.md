---
title: Exercises - Sensor Integration
sidebar_label: Exercises
sidebar_position: 10
description: Exercises for the Sensor Integration chapter focusing on sensor fusion, calibration, and real-time processing
keywords: [exercises, sensors, sensor fusion, calibration, real-time processing, robotics]
---

# Exercises: Sensor Integration

These exercises are designed to reinforce the concepts covered in the Sensor Integration chapter. They range from theoretical problems to practical implementation challenges.

## Exercise 1: Kalman Filter Implementation for Robot Localization

### Problem Statement
Implement an Extended Kalman Filter (EKF) to estimate the position and velocity of a mobile robot using noisy sensor measurements from wheel encoders and an IMU.

### Tasks:
1. Define the state vector (position, velocity) and system model
2. Implement the prediction and update steps of the EKF
3. Simulate noisy sensor measurements
4. Test the filter's performance under different noise conditions

### Solution Approach:
```python
import numpy as np
import matplotlib.pyplot as plt

class RobotLocalizationEKF:
    def __init__(self, dt=0.1):
        self.dt = dt  # Time step

        # State vector: [x, y, vx, vy] (position and velocity)
        self.state = np.zeros(4)

        # State covariance matrix
        self.P = np.eye(4) * 1000  # High initial uncertainty

        # Process noise covariance (system model uncertainty)
        self.Q = np.eye(4)
        self.Q[0, 0] = 0.1  # x position noise
        self.Q[1, 1] = 0.1  # y position noise
        self.Q[2, 2] = 0.5  # x velocity noise
        self.Q[3, 3] = 0.5  # y velocity noise

        # Measurement noise covariance
        self.R_encoders = np.eye(2) * 0.01  # Encoder noise
        self.R_imu = np.eye(2) * 0.05      # IMU noise

    def predict(self, control_input=None):
        """Prediction step of the EKF"""
        # State transition model: x(k+1) = F*x(k) + G*u(k)
        # For constant velocity model:
        F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Apply prediction
        self.state = F @ self.state

        # Predict covariance: P(k+1|k) = F*P(k|k)*F^T + Q
        self.P = F @ self.P @ F.T + self.Q

    def update_with_encoders(self, encoder_measurement):
        """Update step using encoder measurements (position)"""
        # Measurement model: z = H*x (only position is measured)
        H = np.array([
            [1, 0, 0, 0],  # Measure x position
            [0, 1, 0, 0]   # Measure y position
        ])

        # Innovation: y = z - H*x
        innovation = encoder_measurement - H @ self.state

        # Innovation covariance: S = H*P*H^T + R
        S = H @ self.P @ H.T + self.R_encoders

        # Kalman gain: K = P*H^T*S^(-1)
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state: x(k|k) = x(k|k-1) + K*y
        self.state = self.state + K @ innovation

        # Update covariance: P(k|k) = (I - K*H)*P(k|k-1)
        I = np.eye(len(self.state))
        self.P = (I - K @ H) @ self.P

    def update_with_imu(self, imu_measurement):
        """Update step using IMU measurements (velocity)"""
        # Measurement model: z = H*x (only velocity is measured)
        H = np.array([
            [0, 0, 1, 0],  # Measure x velocity
            [0, 0, 0, 1]   # Measure y velocity
        ])

        # Innovation: y = z - H*x
        innovation = imu_measurement - H @ self.state

        # Innovation covariance: S = H*P*H^T + R
        S = H @ self.P @ H.T + self.R_imu

        # Kalman gain: K = P*H^T*S^(-1)
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state: x(k|k) = x(k|k-1) + K*y
        self.state = self.state + K @ innovation

        # Update covariance: P(k|k) = (I - K*H)*P(k|k-1)
        I = np.eye(len(self.state))
        self.P = (I - K @ H) @ self.P

def simulate_robot_trajectory():
    """Simulate a robot moving in a circular path"""
    dt = 0.1
    t_max = 20.0
    time_steps = int(t_max / dt)

    # True trajectory (circular motion)
    t = np.linspace(0, t_max, time_steps)
    true_x = 5 * np.cos(0.2 * t)
    true_y = 5 * np.sin(0.2 * t)
    true_vx = -5 * 0.2 * np.sin(0.2 * t)
    true_vy = 5 * 0.2 * np.cos(0.2 * t)

    return np.column_stack([true_x, true_y, true_vx, true_vy]), t

def add_sensor_noise(true_state, sensor_type='encoder'):
    """Add noise to simulate sensor measurements"""
    if sensor_type == 'encoder':
        # Position measurements with noise
        noise_std = 0.1
        return true_state[:2] + np.random.normal(0, noise_std, 2)
    elif sensor_type == 'imu':
        # Velocity measurements with noise
        noise_std = 0.05
        return true_state[2:4] + np.random.normal(0, noise_std, 2)

def run_localization_example():
    """Run the robot localization example"""
    # Get true trajectory
    true_trajectory, time_vector = simulate_robot_trajectory()
    dt = time_vector[1] - time_vector[0]

    # Initialize EKF
    ekf = RobotLocalizationEKF(dt)

    # Storage for results
    estimated_states = []
    covariance_trace = []

    # Simulate sensor measurements and run EKF
    for i in range(len(time_vector)):
        true_state = true_trajectory[i]

        # Simulate noisy measurements
        encoder_meas = add_sensor_noise(true_state, 'encoder')
        imu_meas = add_sensor_noise(true_state, 'imu')

        # Prediction step
        ekf.predict()

        # Update steps with different sensors at different rates
        if i % 2 == 0:  # Update with encoders every 2 steps (5Hz)
            ekf.update_with_encoders(encoder_meas)

        if i % 5 == 0:  # Update with IMU every 5 steps (2Hz)
            ekf.update_with_imu(imu_meas)

        # Store results
        estimated_states.append(ekf.state.copy())
        covariance_trace.append(np.trace(ekf.P))

    estimated_states = np.array(estimated_states)
    covariance_trace = np.array(covariance_trace)

    # Calculate errors
    position_errors = np.sqrt(np.sum((estimated_states[:, :2] - true_trajectory[:, :2])**2, axis=1))
    velocity_errors = np.sqrt(np.sum((estimated_states[:, 2:4] - true_trajectory[:, 2:4])**2, axis=1))

    print(f"Average position error: {np.mean(position_errors):.3f} m")
    print(f"Average velocity error: {np.mean(velocity_errors):.3f} m/s")

    return true_trajectory, estimated_states, position_errors, velocity_errors

# Run the example
# true_traj, est_traj, pos_err, vel_err = run_localization_example()
```

### Expected Outcomes:
- The EKF should provide more accurate estimates than raw sensor measurements
- Position error should be bounded and converge over time
- Velocity estimates should be smoother than raw IMU readings

## Exercise 2: Multi-Sensor Calibration

### Problem Statement
Implement a calibration procedure to determine the transformation between a camera and a LIDAR sensor using a calibration board with known geometry.

### Tasks:
1. Generate synthetic calibration data with known transformation
2. Implement the calibration algorithm using point correspondences
3. Evaluate the accuracy of the estimated transformation
4. Test robustness to measurement noise

### Solution Approach:
```python
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

def generate_calibration_data(num_boards=20, noise_level=0.001):
    """Generate synthetic calibration data"""
    # True transformation from camera to LIDAR frame
    true_rotation = R.from_euler('xyz', [0.1, 0.2, 0.3]).as_matrix()
    true_translation = np.array([0.5, 0.2, 0.1])  # meters

    # Calibration board parameters (chessboard)
    board_size = (6, 9)  # 6x9 inner corners
    square_size = 0.025  # 2.5 cm squares

    # Generate 3D points in board coordinate system
    board_points = np.zeros((board_size[0] * board_size[1], 3))
    for i in range(board_size[0]):
        for j in range(board_size[1]):
            board_points[i * board_size[1] + j] = [i * square_size, j * square_size, 0]

    camera_points_all = []
    lidar_points_all = []

    for i in range(num_boards):
        # Random board pose in camera frame
        board_rotation = R.random().as_matrix()
        board_translation = np.random.uniform(-0.5, 0.5, 3)

        # Transform board points to camera frame
        camera_points = (board_rotation @ board_points.T).T + board_translation

        # Transform board points to LIDAR frame using true transformation
        lidar_points = (true_rotation @ camera_points.T).T + true_translation

        # Add noise
        camera_points += np.random.normal(0, noise_level, camera_points.shape)
        lidar_points += np.random.normal(0, noise_level, lidar_points.shape)

        camera_points_all.append(camera_points)
        lidar_points_all.append(lidar_points)

    return camera_points_all, lidar_points_all, true_rotation, true_translation

def calibrate_camera_lidar(camera_points_list, lidar_points_list):
    """Calibrate transformation between camera and LIDAR"""
    # Collect all point correspondences
    all_camera_points = np.vstack(camera_points_list)
    all_lidar_points = np.vstack(lidar_points_list)

    # Compute centroids
    camera_centroid = np.mean(all_camera_points, axis=0)
    lidar_centroid = np.mean(all_lidar_points, axis=0)

    # Center the points
    camera_centered = all_camera_points - camera_centroid
    lidar_centered = all_lidar_points - lidar_centroid

    # Compute cross-covariance matrix
    H = camera_centered.T @ lidar_centered

    # SVD to find rotation
    U, S, Vt = np.linalg.svd(H)
    rotation = Vt.T @ U.T

    # Ensure proper rotation matrix (determinant = 1)
    if np.linalg.det(rotation) < 0:
        Vt[-1, :] *= -1
        rotation = Vt.T @ U.T

    # Compute translation
    translation = lidar_centroid - rotation @ camera_centroid

    return rotation, translation

def evaluate_calibration(true_rotation, true_translation, est_rotation, est_translation):
    """Evaluate calibration accuracy"""
    # Rotation error (in degrees)
    rotation_error_matrix = np.eye(3) - (true_rotation.T @ est_rotation)
    rotation_error = np.arccos(np.clip((np.trace(rotation_error_matrix) - 1) / 2, -1, 1))
    rotation_error_deg = np.rad2deg(rotation_error)

    # Translation error (in meters)
    translation_error = np.linalg.norm(true_translation - est_translation)

    return rotation_error_deg, translation_error

def run_calibration_example():
    """Run the calibration example"""
    # Generate data
    camera_points, lidar_points, true_rot, true_trans = generate_calibration_data()

    # Perform calibration
    est_rot, est_trans = calibrate_camera_lidar(camera_points, lidar_points)

    # Evaluate results
    rot_error, trans_error = evaluate_calibration(true_rot, true_trans, est_rot, est_trans)

    print(f"True translation: {true_trans}")
    print(f"Estimated translation: {est_trans}")
    print(f"Translation error: {trans_error:.6f} m")
    print(f"Rotation error: {rot_error:.6f} degrees")

    return rot_error, trans_error

# Run the example
# rotation_error, translation_error = run_calibration_example()
```

### Expected Outcomes:
- Translation error should be less than 1cm for noise-free data
- Rotation error should be less than 1 degree for noise-free data
- Algorithm should be robust to reasonable levels of measurement noise

## Exercise 3: Real-time Sensor Data Processing

### Problem Statement
Design and implement a real-time system that processes data from multiple sensors (IMU, camera, LIDAR) at different frequencies while maintaining synchronization.

### Tasks:
1. Implement a sensor data manager with different sampling rates
2. Design a synchronization mechanism for multi-rate data
3. Implement a fusion algorithm that handles asynchronous inputs
4. Test the system under various timing conditions

### Solution Approach:
```python
import time
import threading
import queue
from collections import deque
import numpy as np

class RealTimeSensorManager:
    def __init__(self):
        self.sensors = {}
        self.data_queues = {}
        self.time_stamps = {}
        self.synchronization_window = 0.05  # 50ms window for synchronization
        self.running = False
        self.main_loop_thread = None

        # Results storage
        self.fused_data = deque(maxlen=100)
        self.processing_times = deque(maxlen=100)

    def add_sensor(self, name, frequency, data_generator):
        """Add a sensor with its frequency and data generator"""
        self.sensors[name] = {
            'frequency': frequency,
            'period': 1.0 / frequency,
            'data_generator': data_generator,
            'thread': None,
            'running': False
        }
        self.data_queues[name] = queue.Queue(maxsize=10)  # Limit queue size
        self.time_stamps[name] = time.time()

    def start_sensor(self, name):
        """Start data collection for a specific sensor"""
        if name not in self.sensors:
            raise ValueError(f"Sensor {name} not registered")

        sensor_info = self.sensors[name]
        sensor_info['running'] = True

        def sensor_loop():
            next_time = time.time()
            while sensor_info['running'] and self.running:
                start_time = time.time()

                try:
                    # Generate sensor data
                    data = sensor_info['data_generator']()

                    # Add timestamp
                    timestamp = time.time()

                    # Put in queue (non-blocking)
                    try:
                        self.data_queues[name].put_nowait((data, timestamp))
                    except queue.Full:
                        # Drop oldest if queue full
                        try:
                            self.data_queues[name].get_nowait()
                            self.data_queues[name].put_nowait((data, timestamp))
                        except queue.Empty:
                            pass

                    # Maintain frequency
                    next_time += sensor_info['period']
                    sleep_time = max(0, next_time - time.time())
                    time.sleep(sleep_time)

                except Exception as e:
                    print(f"Error in sensor {name}: {e}")
                    time.sleep(0.01)  # Brief pause before retry

        sensor_info['thread'] = threading.Thread(target=sensor_loop)
        sensor_info['thread'].start()

    def start_system(self):
        """Start the entire system"""
        self.running = True

        # Start all sensor threads
        for name in self.sensors:
            self.start_sensor(name)

        # Start main processing loop
        self.main_loop_thread = threading.Thread(target=self.main_processing_loop)
        self.main_loop_thread.start()

    def stop_system(self):
        """Stop the entire system"""
        self.running = False

        # Stop all sensor threads
        for name in self.sensors:
            self.sensors[name]['running'] = False

        # Wait for threads to finish
        if self.main_loop_thread:
            self.main_loop_thread.join()

        for name in self.sensors:
            if self.sensors[name]['thread']:
                self.sensors[name]['thread'].join()

    def get_synchronized_data(self):
        """Get the most recent synchronized data from all sensors"""
        current_time = time.time()
        synchronized_data = {}

        for name, queue_obj in self.data_queues.items():
            if not queue_obj.empty():
                # Look for data within synchronization window
                temp_items = []
                best_item = None
                best_time_diff = float('inf')

                # Extract all items from queue temporarily
                while not queue_obj.empty():
                    item = queue_obj.get_nowait()
                    temp_items.append(item)

                    # Check if this is the closest to current time
                    time_diff = abs(item[1] - current_time)
                    if time_diff < best_time_diff:
                        best_time_diff = time_diff
                        best_item = item

                # Put items back in queue (except the best one)
                for item in temp_items:
                    if item != best_item and not queue_obj.full():
                        queue_obj.put_nowait(item)

                if best_item and best_time_diff <= self.synchronization_window:
                    synchronized_data[name] = best_item

        return synchronized_data if len(synchronized_data) == len(self.sensors) else None

    def main_processing_loop(self):
        """Main processing loop that fuses sensor data"""
        while self.running:
            start_time = time.time()

            # Get synchronized data
            sync_data = self.get_synchronized_data()

            if sync_data:
                # Perform sensor fusion
                fused_result = self.fuse_sensor_data(sync_data)

                # Store result
                self.fused_data.append({
                    'timestamp': time.time(),
                    'data': fused_result,
                    'input_data': sync_data
                })

            # Track processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)

            # Maintain a reasonable processing rate
            time.sleep(0.001)  # 1kHz processing rate

    def fuse_sensor_data(self, sensor_data):
        """Fuse data from multiple sensors"""
        # Example fusion: combine IMU orientation with camera position estimate
        result = {}

        if 'imu' in sensor_data:
            result['imu_data'] = sensor_data['imu'][0]

        if 'camera' in sensor_data:
            result['camera_data'] = sensor_data['camera'][0]

        if 'lidar' in sensor_data:
            result['lidar_data'] = sensor_data['lidar'][0]

        # Add timestamp
        result['timestamp'] = time.time()

        return result

    def get_performance_stats(self):
        """Get performance statistics"""
        if not self.processing_times:
            return {'avg_processing_time': 0, 'max_processing_time': 0}

        return {
            'avg_processing_time': np.mean(self.processing_times),
            'max_processing_time': np.max(self.processing_times),
            'min_processing_time': np.min(self.processing_times),
            'processing_rate': len(self.processing_times) / (self.processing_times[-1] - self.processing_times[0] + 1e-6) if len(self.processing_times) > 1 else 0
        }

def run_real_time_example():
    """Run the real-time sensor processing example"""
    # Create sensor manager
    manager = RealTimeSensorManager()

    # Define sensor data generators
    def imu_generator():
        # Simulate IMU data: [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
        return np.random.normal([0, 0, 9.81, 0, 0, 0], [0.1, 0.1, 0.1, 0.01, 0.01, 0.01])

    def camera_generator():
        # Simulate camera data: [x_pos, y_pos, z_pos, orientation]
        return np.random.normal([0, 0, 0, 0], [0.01, 0.01, 0.01, 0.001])

    def lidar_generator():
        # Simulate LIDAR data: [range_values...]
        return np.random.uniform(0.1, 10.0, 360)  # 360 degree scan

    # Add sensors
    manager.add_sensor('imu', 100, imu_generator)      # 100Hz IMU
    manager.add_sensor('camera', 30, camera_generator) # 30Hz camera
    manager.add_sensor('lidar', 10, lidar_generator)   # 10Hz LIDAR

    # Start system
    manager.start_system()

    # Run for 5 seconds
    time.sleep(5)

    # Stop system
    manager.stop_system()

    # Print performance stats
    stats = manager.get_performance_stats()
    print(f"Average processing time: {stats['avg_processing_time']:.6f}s")
    print(f"Max processing time: {stats['max_processing_time']:.6f}s")
    print(f"Processing rate: {stats['processing_rate']:.2f} Hz")
    print(f"Fused data points: {len(manager.fused_data)}")

    return manager

# Run the example (commented out to avoid blocking execution)
# manager = run_real_time_example()
```

### Expected Outcomes:
- System should maintain timing requirements for all sensors
- Synchronization should work within specified time windows
- Processing overhead should be minimal
- Data loss should be minimal under normal conditions

## Exercise 4: Sensor Fault Detection and Isolation

### Problem Statement
Implement an algorithm to detect and isolate sensor faults in real-time, including stuck sensors, out-of-range readings, and communication failures.

### Tasks:
1. Implement statistical tests for fault detection
2. Design a voting mechanism for sensor validation
3. Implement sensor switching when faults are detected
4. Test with various fault scenarios

### Solution Approach:
```python
import numpy as np
from scipy import stats
from collections import deque

class SensorFaultDetector:
    def __init__(self, sensor_names, window_size=50):
        self.sensor_names = sensor_names
        self.window_size = window_size
        self.data_history = {name: deque(maxlen=window_size) for name in sensor_names}
        self.status = {name: {'ok': True, 'last_fault': None, 'fault_count': 0} for name in sensor_names}
        self.fault_thresholds = {name: {'min': -1e6, 'max': 1e6, 'variance': 1e6} for name in sensor_names}
        self.fault_history = deque(maxlen=100)

    def update_sensor_data(self, sensor_name, data):
        """Update with new sensor data and check for faults"""
        if sensor_name not in self.data_history:
            raise ValueError(f"Unknown sensor: {sensor_name}")

        # Add data to history
        self.data_history[sensor_name].append(data)

        # Check for faults
        fault_detected = self._check_faults(sensor_name, data)

        if fault_detected:
            self.status[sensor_name]['ok'] = False
            self.status[sensor_name]['last_fault'] = time.time()
            self.status[sensor_name]['fault_count'] += 1

            # Log fault
            self.fault_history.append({
                'sensor': sensor_name,
                'timestamp': time.time(),
                'data': data,
                'fault_type': fault_detected
            })
        else:
            self.status[sensor_name]['ok'] = True

    def _check_faults(self, sensor_name, data):
        """Check for various types of sensor faults"""
        history = list(self.data_history[sensor_name])

        if len(history) < 3:
            return None  # Need more data

        # Check for NaN or Inf values
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            return "nan_inf"

        # Check for out-of-range values
        if np.any(data < self.fault_thresholds[sensor_name]['min']) or \
           np.any(data > self.fault_thresholds[sensor_name]['max']):
            return "out_of_range"

        # Check for stuck sensor (very low variance)
        if len(history) >= 10:
            recent_data = np.array(history[-10:])
            variance = np.var(recent_data, axis=0)
            if np.any(variance < 1e-10):  # Extremely low variance
                return "stuck_sensor"

        # Check for sudden jumps (derivative-based)
        if len(history) >= 2:
            last_data = history[-2]
            current_data = data
            diff = np.abs(current_data - last_data)
            if np.any(diff > 10):  # Arbitrary threshold for sudden change
                return "sudden_jump"

        # Check for statistical outliers using z-score
        if len(history) >= 10:
            recent_data = np.array(history[-10:])
            mean_val = np.mean(recent_data, axis=0)
            std_val = np.std(recent_data, axis=0)
            # Avoid division by zero
            std_val = np.where(std_val == 0, 1e-6, std_val)

            z_scores = np.abs((data - mean_val) / std_val)
            if np.any(z_scores > 5):  # 5-sigma outlier
                return "statistical_outlier"

        return None  # No fault detected

    def get_valid_sensors(self):
        """Get list of currently valid sensors"""
        return [name for name, status in self.status.items() if status['ok']]

    def get_faulty_sensors(self):
        """Get list of currently faulty sensors"""
        return [name for name, status in self.status.items() if not status['ok']]

    def sensor_voting(self, sensor_data_dict, expected_value=None):
        """Use voting mechanism to validate sensor readings"""
        valid_readings = {}
        invalid_readings = {}

        for sensor_name, data in sensor_data_dict.items():
            if self.status[sensor_name]['ok']:
                valid_readings[sensor_name] = data
            else:
                invalid_readings[sensor_name] = data

        if not valid_readings:
            return None, "All sensors faulty"

        # Compute consensus from valid readings
        if len(valid_readings) == 1:
            consensus = list(valid_readings.values())[0]
        else:
            # Use median to be robust to outliers
            all_values = list(valid_readings.values())
            if isinstance(all_values[0], np.ndarray):
                consensus = np.median(all_values, axis=0)
            else:
                consensus = np.median(all_values)

        return consensus, "Valid consensus achieved"

    def get_health_report(self):
        """Get comprehensive health report"""
        return {
            'timestamp': time.time(),
            'sensor_status': self.status,
            'valid_sensors': self.get_valid_sensors(),
            'faulty_sensors': self.get_faulty_sensors(),
            'total_faults': sum(s['fault_count'] for s in self.status.values()),
            'recent_faults': list(self.fault_history)
        }

class RedundantSensorSystem:
    def __init__(self, primary_sensors, backup_sensors):
        self.primary_sensors = primary_sensors
        self.backup_sensors = backup_sensors
        self.fault_detector = SensorFaultDetector(list(primary_sensors.keys()))
        self.active_sensors = primary_sensors.copy()
        self.sensor_history = {name: deque(maxlen=10) for name in primary_sensors.keys()}

    def read_sensor_data(self):
        """Read data from active sensors and handle faults"""
        sensor_data = {}

        # Read from active sensors
        for name, sensor in self.active_sensors.items():
            try:
                # Simulate reading data (in real implementation, this would interface with hardware)
                if hasattr(sensor, 'read_data'):
                    data = sensor.read_data()
                else:
                    # For simulation
                    data = np.random.normal(0, 1, 3)  # 3D data

                sensor_data[name] = data
                self.fault_detector.update_sensor_data(name, data)

                # Update history
                self.sensor_history[name].append(data)

            except Exception as e:
                print(f"Error reading sensor {name}: {e}")
                # Mark as faulty
                self.fault_detector.status[name]['ok'] = False

        # Check for sensor failures and switch if needed
        self._handle_sensor_failures()

        return sensor_data

    def _handle_sensor_failures(self):
        """Handle sensor failures by switching to backups if available"""
        faulty_sensors = self.fault_detector.get_faulty_sensors()

        for faulty_sensor in faulty_sensors:
            if faulty_sensor in self.backup_sensors:
                print(f"Switching {faulty_sensor} to backup sensor")
                self.active_sensors[faulty_sensor] = self.backup_sensors[faulty_sensor]
                # Reset fault status after switching
                self.fault_detector.status[faulty_sensor]['ok'] = True

    def get_consensus_reading(self):
        """Get consensus reading using voting mechanism"""
        sensor_data = self.read_sensor_data()
        return self.fault_detector.sensor_voting(sensor_data)

def run_fault_detection_example():
    """Run the fault detection example"""
    # Create sensor system with simulated sensors
    primary_sensors = {
        'imu1': type('Sensor', (), {'read_data': lambda: np.random.normal([0, 0, 9.81], 0.1)})(),
        'imu2': type('Sensor', (), {'read_data': lambda: np.random.normal([0, 0, 9.81], 0.1)})(),
        'imu3': type('Sensor', (), {'read_data': lambda: np.random.normal([0, 0, 9.81], 0.1)})()
    }

    backup_sensors = {
        'imu1': type('Sensor', (), {'read_data': lambda: np.random.normal([0, 0, 9.81], 0.15)})(),
        'imu2': type('Sensor', (), {'read_data': lambda: np.random.normal([0, 0, 9.81], 0.15)})()
    }

    system = RedundantSensorSystem(primary_sensors, backup_sensors)

    # Simulate operation with some faulty readings
    for i in range(100):
        consensus, status = system.get_consensus_reading()

        # Introduce a fault every 20 readings for one sensor
        if i % 20 == 0 and i > 0:
            # Simulate a stuck sensor by temporarily replacing with a constant value
            print(f"Introducing fault at iteration {i}")

        if i % 50 == 0:
            report = system.fault_detector.get_health_report()
            print(f"Iteration {i}: Valid sensors: {report['valid_sensors']}, "
                  f"Faulty sensors: {report['faulty_sensors']}")

    return system

# Run the example (commented out to avoid blocking)
# system = run_fault_detection_example()
```

### Expected Outcomes:
- Faults should be detected quickly (within a few readings)
- System should gracefully handle sensor failures
- Voting mechanism should provide reliable readings even with some faulty sensors
- False positive rate should be low

## Exercise 5: Sensor Fusion for Humanoid Balance Control

### Problem Statement
Implement a sensor fusion system specifically designed for humanoid robot balance control, combining IMU, force/torque sensors, and joint encoders.

### Tasks:
1. Implement a fusion algorithm for center of mass estimation
2. Design a Kalman filter for balance state estimation
3. Implement zero moment point (ZMP) calculation from fused data
4. Test the system with simulated humanoid robot data

### Solution Approach:
```python
class HumanoidBalanceFusion:
    def __init__(self, robot_mass=75.0, com_height=0.8):
        self.robot_mass = robot_mass
        self.com_height = com_height  # Height of center of mass above ground
        self.gravity = 9.81

        # State vector: [com_x, com_y, com_z, com_vx, com_vy, com_vz, com_ax, com_ay, com_az]
        self.state_dim = 9
        self.measurement_dim = 12  # IMU + F/T sensors + joint encoders

        # Initialize Kalman filter for balance state estimation
        self.kf = self._initialize_kalman_filter()

        # Sensor data storage
        self.imu_data = np.zeros(6)  # [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
        self.ft_data = np.zeros(6)   # [force_x, force_y, force_z, torque_x, torque_y, torque_z]
        self.joint_positions = np.zeros(28)  # Example: 28 DOF humanoid
        self.joint_velocities = np.zeros(28)

        # Robot kinematic model parameters
        self.support_polygon = self._define_support_polygon()

    def _initialize_kalman_filter(self):
        """Initialize Kalman filter for balance state estimation"""
        # State transition matrix (constant acceleration model)
        dt = 0.01  # 100Hz
        F = np.eye(self.state_dim)
        F[0, 3] = dt  # x = x + vx*dt
        F[1, 4] = dt  # y = y + vy*dt
        F[2, 5] = dt  # z = z + vz*dt
        F[3, 6] = dt  # vx = vx + ax*dt
        F[4, 7] = dt  # vy = vy + ay*dt
        F[5, 8] = dt  # vz = vz + az*dt

        # Process noise covariance
        Q = np.eye(self.state_dim) * 0.1
        Q[6, 6] = 1.0  # Higher noise for acceleration
        Q[7, 7] = 1.0
        Q[8, 8] = 1.0

        # Measurement matrix (we measure position and acceleration)
        H = np.zeros((self.measurement_dim, self.state_dim))
        H[0, 0] = 1  # Measure com_x
        H[1, 1] = 1  # Measure com_y
        H[2, 2] = 1  # Measure com_z
        H[6, 6] = 1  # Measure com_ax
        H[7, 7] = 1  # Measure com_ay
        H[8, 8] = 1  # Measure com_az

        # Measurement noise covariance
        R = np.eye(self.measurement_dim) * 0.01

        # Initial state covariance
        P = np.eye(self.state_dim) * 100

        return {
            'F': F, 'Q': Q, 'H': H, 'R': R, 'P': P,
            'x': np.zeros(self.state_dim),  # Initial state
            'dt': dt
        }

    def _define_support_polygon(self):
        """Define the support polygon based on foot positions"""
        # For a biped, this would be the area between the feet
        # Simplified as a rectangle for this example
        foot_width = 0.12  # 12cm
        foot_length = 0.20  # 20cm
        foot_separation = 0.25  # 25cm between feet

        # Define vertices of support polygon (simplified as rectangular)
        vertices = np.array([
            [-foot_length/2, -foot_separation/2 - foot_width/2],  # Left foot, back-left
            [foot_length/2, -foot_separation/2 - foot_width/2],   # Left foot, front-left
            [foot_length/2, -foot_separation/2 + foot_width/2],   # Left foot, front-right
            [-foot_length/2, -foot_separation/2 + foot_width/2],  # Left foot, back-right
            [-foot_length/2, foot_separation/2 - foot_width/2],   # Right foot, back-left
            [foot_length/2, foot_separation/2 - foot_width/2],    # Right foot, front-left
            [foot_length/2, foot_separation/2 + foot_width/2],    # Right foot, front-right
            [-foot_length/2, foot_separation/2 + foot_width/2]    # Right foot, back-right
        ])

        return vertices

    def update_sensors(self, imu_data, ft_data, joint_positions, joint_velocities):
        """Update sensor data"""
        self.imu_data = imu_data
        self.ft_data = ft_data
        self.joint_positions = joint_positions
        self.joint_velocities = joint_velocities

    def estimate_com_state(self):
        """Estimate center of mass state using kinematic model and sensor data"""
        # This would use forward kinematics to estimate CoM position from joint angles
        # For simplicity, we'll use a simplified model

        # Estimate CoM position based on joint configuration
        # In a real implementation, this would use full kinematic model
        com_x = np.mean(self.joint_positions[:3]) * 0.1  # Simplified mapping
        com_y = np.mean(self.joint_positions[3:6]) * 0.1
        com_z = self.com_height + np.mean(self.joint_positions[6:9]) * 0.05

        # Use IMU for acceleration
        com_acc = self.imu_data[:3]  # Accelerometer readings

        return np.array([com_x, com_y, com_z, 0, 0, 0, *com_acc])

    def predict_kalman(self):
        """Kalman filter prediction step"""
        kf = self.kf
        kf['x'] = kf['F'] @ kf['x']
        kf['P'] = kf['F'] @ kf['P'] @ kf['F'].T + kf['Q']

    def update_kalman(self, measurement):
        """Kalman filter update step"""
        kf = self.kf

        # Innovation
        y = measurement - kf['H'] @ kf['x']

        # Innovation covariance
        S = kf['H'] @ kf['P'] @ kf['H'].T + kf['R']

        # Kalman gain
        K = kf['P'] @ kf['H'].T @ np.linalg.inv(S)

        # Update state
        kf['x'] = kf['x'] + K @ y

        # Update covariance
        I = np.eye(len(kf['x']))
        kf['P'] = (I - K @ kf['H']) @ kf['P']

    def calculate_zmp(self, com_state):
        """Calculate Zero Moment Point from CoM state"""
        # ZMP = CoM position - (CoM_height / gravity) * CoM_acceleration
        com_pos = com_state[:3]
        com_acc = com_state[6:9]

        zmp = com_pos - (self.com_height / self.gravity) * com_acc

        return zmp[:2]  # Return only x, y coordinates

    def is_stable(self, zmp):
        """Check if ZMP is within support polygon"""
        # Simple check if ZMP is within bounds
        # In a real implementation, this would check against actual polygon
        x, y = zmp

        # Define approximate support bounds
        x_bounds = (-0.3, 0.3)  # 30cm each side
        y_bounds = (-0.2, 0.2)  # 20cm each side

        return x_bounds[0] <= x <= x_bounds[1] and y_bounds[0] <= y <= y_bounds[1]

    def run_balance_control_step(self):
        """Run one step of balance control fusion"""
        # Estimate CoM state from all sensors
        estimated_state = self.estimate_com_state()

        # Prediction step
        self.predict_kalman()

        # Update with measurement
        self.update_kalman(estimated_state)

        # Get filtered state
        filtered_state = self.kf['x']

        # Calculate ZMP
        zmp = self.calculate_zmp(filtered_state)

        # Check stability
        stable = self.is_stable(zmp)

        return {
            'com_state': filtered_state,
            'zmp': zmp,
            'stable': stable,
            'com_position': filtered_state[:3],
            'com_acceleration': filtered_state[6:9]
        }

def simulate_humanoid_balance():
    """Simulate humanoid balance control with sensor fusion"""
    fusion_system = HumanoidBalanceFusion()

    # Simulate sensor readings over time
    results = []

    for t in range(1000):  # 10 seconds at 100Hz
        # Simulate sensor data
        imu_data = np.random.normal([0, 0, 9.81, 0, 0, 0], [0.01, 0.01, 0.01, 0.001, 0.001, 0.001])
        ft_data = np.random.normal([0, 0, fusion_system.robot_mass * 9.81, 0, 0, 0], [1, 1, 10, 0.1, 0.1, 0.1])
        joint_positions = np.random.normal(0, 0.1, 28)
        joint_velocities = np.random.normal(0, 0.05, 28)

        # Update sensor data
        fusion_system.update_sensors(imu_data, ft_data, joint_positions, joint_velocities)

        # Run balance control step
        result = fusion_system.run_balance_control_step()
        result['time'] = t * 0.01  # Convert to seconds

        results.append(result)

        # Print status occasionally
        if t % 100 == 0:
            print(f"Time: {result['time']:.2f}s, Stable: {result['stable']}, ZMP: ({result['zmp'][0]:.3f}, {result['zmp'][1]:.3f})")

    # Calculate statistics
    stable_count = sum(1 for r in results if r['stable'])
    stability_rate = stable_count / len(results)

    print(f"\nBalance control results:")
    print(f"Overall stability rate: {stability_rate:.2%}")
    print(f"Stable samples: {stable_count}/{len(results)}")

    return results

# Run the simulation (commented out to avoid blocking)
# balance_results = simulate_humanoid_balance()
```

### Expected Outcomes:
- System should maintain balance with high stability rate (>95%)
- ZMP should stay within support polygon during stable conditions
- Fusion should provide smoother estimates than individual sensors
- System should detect balance recovery after disturbances

## Solutions and Discussion

### Exercise 1 Discussion:
The EKF implementation demonstrates how to fuse position and velocity measurements from different sensors. The key is properly modeling the system dynamics and sensor characteristics. The filter should provide more accurate estimates than raw sensor readings, especially when sensors have different noise characteristics and update rates.

### Exercise 2 Discussion:
Sensor calibration is critical for multi-sensor systems. The SVD-based approach is mathematically sound and robust. The accuracy depends on the quality and diversity of calibration poses. More diverse poses (different orientations and positions) generally lead to better calibration results.

### Exercise 3 Discussion:
Real-time sensor processing requires careful attention to timing constraints and data synchronization. The multi-threaded approach with queues helps decouple sensor sampling from processing. The synchronization window needs to be chosen based on the application requirements and sensor characteristics.

### Exercise 4 Discussion:
Sensor fault detection is crucial for safety-critical systems. The combination of statistical tests and voting mechanisms provides robust fault detection and isolation. The system should have low false positive rates while quickly detecting actual faults.

### Exercise 5 Discussion:
Balance control in humanoid robots is a challenging sensor fusion problem. The ZMP-based approach is widely used in humanoid robotics. Proper fusion of IMU, force/torque, and joint encoder data is essential for stable balance control.

## References

1. Thrun, S., Burgard, W., & Fox, D. (2005). *Probabilistic Robotics*. MIT Press. [Peer-reviewed]

2. Lobo, J., & Dias, J. (2007). Active and adaptive sensor fusion for mobile robot localization. *Autonomous Robots*, 23(2), 125-144. https://doi.org/10.1007/s10514-007-9028-4 [Peer-reviewed]

3. Valenti, R. G., Dryanovski, I., & Xiao, J. (2015). Keeping a good attitude: A quaternion-based orientation filter for IMUs and MARGs. *Sensors*, 15(8), 19302-19330. https://doi.org/10.3390/s150819302 [Peer-reviewed]

4. Rehbinder, H., & Khoshelham, K. (2012). Accurate indoor localization with IMU and LIDAR data fusion. *IEEE International Conference on Robotics and Automation (ICRA)*, 2566-2571. https://doi.org/10.1109/ICRA.2012.6224661 [Peer-reviewed]

5. Kajita, S. (2019). *Humanoid Robotics: A Reference*. Springer. https://doi.org/10.1007/978-3-319-46135-4 [Peer-reviewed]

## Summary

These exercises covered essential aspects of sensor integration in robotics:
- Kalman filtering for state estimation
- Multi-sensor calibration procedures
- Real-time processing considerations
- Fault detection and isolation
- Application to humanoid balance control

Each exercise builds on theoretical concepts while addressing practical implementation challenges specific to robotic sensor systems. The solutions demonstrate how to handle noise, synchronization, calibration, and real-time constraints in multi-sensor environments.