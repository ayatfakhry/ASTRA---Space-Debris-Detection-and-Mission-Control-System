# ============================================================
# AEGIS — Default System Configuration
# ============================================================

system:
  name: "AEGIS"
  version: "2.0.0"
  seed: 42
  device: "auto"          # auto | cpu | cuda | mps
  log_level: "INFO"
  output_dir: "outputs/"
  checkpoint_dir: "checkpoints/"

# ── Orbital Simulation ───────────────────────────────────────
simulation:
  dt: 1.0                 # Integration time step (seconds)
  duration: 600.0         # Simulation duration (seconds)
  epoch: "2024-01-01T00:00:00"
  
  earth:
    mu: 3.986004418e14    # Standard gravitational parameter (m^3/s^2)
    radius: 6371000.0     # Mean radius (m)
    J2: 1.08263e-3        # J2 oblateness coefficient
    omega: 7.2921150e-5   # Angular velocity (rad/s)

  regimes:
    LEO:
      altitude_min: 200e3
      altitude_max: 2000e3
      debris_density: "high"
    MEO:
      altitude_min: 2000e3
      altitude_max: 35786e3
      debris_density: "medium"
    GEO:
      altitude: 35786e3
      debris_density: "high"

  debris:
    count: 150
    size_distribution: "power_law"   # power_law | uniform | bimodal
    size_min: 0.01                   # meters
    size_max: 10.0                   # meters
    albedo_range: [0.05, 0.35]
    mass_density: 2700.0             # kg/m^3 (aluminum)
    cd: 2.2                          # Drag coefficient

  space_weather:
    f107: 150.0           # Solar flux index (10^-22 W/m^2/Hz)
    f107a: 150.0          # 81-day average
    ap: 15                # Geomagnetic activity index

# ── Detection Model ─────────────────────────────────────────
detection:
  model: "DebrisNet"
  backbone: "csp_darknet53"
  img_size: [640, 640]
  num_classes: 4
  class_names: ["small_debris", "large_debris", "metallic_fragment", "defunct_satellite"]
  conf_threshold: 0.45
  nms_threshold: 0.45
  anchor_scales: [8, 16, 32]
  
  # Feature Pyramid Network
  fpn_out_channels: 256
  transformer_depth: 2
  transformer_heads: 8

# ── Tracking ────────────────────────────────────────────────
tracking:
  algorithm: "deep_sort"    # ekf | deep_sort | aegis_hybrid
  
  ekf:
    process_noise: 1.0e-4
    measurement_noise: 1.0e-2
    initial_covariance: 1.0
  
  deep_sort:
    max_age: 30
    min_hits: 3
    iou_threshold: 0.3
    max_cosine_distance: 0.3
    nn_budget: 100
    feature_dim: 128
  
  trajectory_lstm:
    input_len: 20           # Historical positions
    pred_len: 30            # Future steps to predict
    hidden_dim: 256
    num_layers: 3
    dropout: 0.1
    attention_heads: 8

# ── Collision Prediction ─────────────────────────────────────
collision:
  alert_distance: 5000.0       # meters — WARNING threshold
  critical_distance: 1000.0    # meters — CRITICAL threshold
  tca_lookahead: 86400.0       # seconds (24 hours)
  monte_carlo_samples: 10000
  hard_body_radius: 5.0        # Combined hard-body radius (m)
  
  # Probability thresholds
  alert_probability: 1e-4
  critical_probability: 1e-3

# ── Sensors ─────────────────────────────────────────────────
sensors:
  lidar:
    enabled: true
    channels: 64
    range_max: 500.0          # meters
    range_noise_sigma: 0.02   # meters
    angular_res_az: 0.1       # degrees
    angular_res_el: 0.2       # degrees
    fps: 10
  
  radar:
    enabled: true
    frequency: 77e9           # Hz (77 GHz)
    range_max: 50000.0        # meters
    range_noise_sigma: 1.0    # meters
    velocity_noise_sigma: 0.1 # m/s
    azimuth_beamwidth: 2.0    # degrees
  
  camera:
    enabled: true
    resolution: [1920, 1080]
    fov: 60.0                 # degrees
    focal_length: 50.0        # mm
    pixel_size: 4.65e-6       # meters
    read_noise: 5.0           # electrons
    dark_current: 0.1         # electrons/pixel/second
    quantum_efficiency: 0.85

  fusion:
    algorithm: "bayesian"     # bayesian | dempster_shafer | kalman
    weights:
      lidar: 0.4
      radar: 0.35
      camera: 0.25

# ── Reinforcement Learning ───────────────────────────────────
rl:
  algorithm: "PPO"
  policy: "MlpPolicy"
  learning_rate: 3e-4
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  total_timesteps: 1_000_000
  
  environment:
    max_dv: 10.0              # m/s — max delta-v per maneuver
    fuel_budget: 100.0        # kg
    planning_horizon: 3600.0  # seconds
    reward_weights:
      collision_avoidance: 100.0
      fuel_efficiency: -1.0
      mission_continuity: 10.0

# ── Training ────────────────────────────────────────────────
training:
  detection:
    epochs: 100
    batch_size: 16
    lr: 1e-3
    lr_schedule: "cosine"
    warmup_epochs: 3
    weight_decay: 5e-4
    momentum: 0.937
    augmentation: true
    mixed_precision: true
  
  trajectory:
    epochs: 200
    batch_size: 64
    lr: 1e-3
    lr_schedule: "plateau"
    patience: 20
    gradient_clip: 1.0

# ── Dashboard ───────────────────────────────────────────────
dashboard:
  host: "0.0.0.0"
  port: 8050
  debug: false
  update_interval: 500        # milliseconds
  max_history_points: 500
  theme: "dark_aerospace"
  
  alerts:
    sound_enabled: true
    email_enabled: false
    log_all_events: true

# ── Performance ─────────────────────────────────────────────
performance:
  num_workers: 4
  pin_memory: true
  compile_model: false        # torch.compile (PyTorch 2.0+)
  onnx_export: false
  tensorrt_optimize: false
  batch_inference: true
  inference_batch_size: 8
