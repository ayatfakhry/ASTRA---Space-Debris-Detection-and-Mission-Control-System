# 🚀 ASTRA - Space Debris Detection and Mission Control System

## Overview

**ASTRA** is an advanced **Space Debris Detection and Mission Control System** designed to enhance spacecraft safety by detecting, tracking, and avoiding space debris using **AI** and **real-time sensor fusion**. The project integrates cutting-edge **Deep Learning** techniques, **LiDAR**, **Radar**, and **GNSS** for accurate and efficient operation in space environments.

The system is designed to detect and predict debris trajectories in real-time, enabling spacecraft to autonomously avoid potential collisions and perform docking safely.

---

## Key Features

### 1. **Space Debris Detection & Tracking**
- Real-time **detection** of space debris using **LiDAR** and **Radar** sensors.
- AI-based **object detection** using models like **YOLOv8** and **Faster R-CNN**.
- **Tracking** of debris trajectories and potential collision risks.

### 2. **AI-Based Collision Avoidance**
- **Reinforcement Learning** models such as **PPO (Proximal Policy Optimization)** and **DQN (Deep Q-Learning)** for autonomous decision-making.
- **Real-time collision avoidance** with the ability to autonomously adjust spacecraft trajectories based on predicted debris motion.

### 3. **Sensor Fusion with Kalman Filter**
- Integration of multiple sensor data (LiDAR, Radar, IMU, GNSS) using **Kalman Filter (KF)** to improve **positioning** and **navigation accuracy**.

### 4. **Mission Control Dashboard**
- **Real-time visualization** of debris trajectories, spacecraft position, velocity, fuel consumption, and collision alerts.
- **Interactive dashboard** built using **React/JSX** and **Plotly/Matplotlib** for live data representation.
- Display of sensor status, AI confidence levels, and debris tracking in 3D space.

### 5. **Spacecraft Simulation**
- **Simulation of orbital dynamics** using **Hill-Clohessy-Wiltshire (HCW)** equations.
- Simulate the effect of **space debris** on spacecraft paths, and adjust for **environmental factors** like **solar radiation** or **space weather**.

---

## Technologies Used

| Technology         | Purpose                                       |
|--------------------|-----------------------------------------------|
| **Python**         | Main programming language                     |
| **TensorFlow** / **PyTorch** | Deep learning frameworks (PPO, DQN)            |
| **YOLOv8**         | Object detection for space debris              |
| **Faster R-CNN**   | Advanced object detection                      |
| **Kalman Filter**  | Sensor fusion and positioning correction      |
| **Matplotlib** / **Plotly** | Data visualization (2D/3D plots)           |
| **React/JSX**      | Mission Control Dashboard UI                  |
| **LiDAR** / **Radar** | For detecting and tracking space debris       |
| **OpenCV**         | Image processing and sensor fusion            |
| **GNSS/RTK**       | Real-time kinematic navigation system         |

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/YOUR_USERNAME/ASTRA_Space_Debris_System.git
   cd ASTRA_Space_Debris_System
