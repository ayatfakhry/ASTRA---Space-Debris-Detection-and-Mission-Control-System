# ============================================================
# ASTRA System Configuration
# ============================================================

system:
  name: "ASTRA"
  version: "2.0.0"
  mode: "full"          # full | detection | simulation | dashboard
  seed: 42
  log_level: "INFO"
  gpu_enabled: true
  num_workers: 4

orbital:
  environment: "LEO"    # LEO | MEO | GEO
  altitude_km: 550.0
  inclination_deg: 53.0
  debris_count: 500
  simulation_duration_s: 3600
  time_step_s: 1.0
  propagator: "SGP4"    # SGP4 | J2 | COWELL
  perturbations:
    j2: true
    j3: true
    j4: true
    atmospheric_drag: true
    solar_radiation: true
    lunar_gravity: false

detection:
  model: "yolov8"       # yolov8 | faster_rcnn | hybrid
  confidence_threshold: 0.45
  nms_threshold: 0.35
  input_size: [640, 640]
  classes:
    0: "small_debris"
    1: "large_debris"
    2: "metallic_fragment"
    3: "defunct_satellite"
  checkpoint: "checkpoints/detector_best.pt"

tracking:
  algorithm: "deep_sort"  # deep_sort | sort | byte_track
  max_age: 30
  min_hits: 3
  iou_threshold: 0.3
  max_cosine_distance: 0.4
  ekf:
    process_noise: 0.01
    measurement_noise: 0.1
    initial_covariance: 10.0

prediction:
  model: "transformer"   # lstm | transformer | hybrid
  sequence_length: 30
  prediction_horizon: 120
  hidden_dim: 256
  num_heads: 8
  num_layers: 6
  checkpoint: "checkpoints/predictor_best.pt"

collision:
  method: "monte_carlo"
  num_samples: 10000
  time_horizon_h: 72
  alert_thresholds:
    green: 1.0e-5
    yellow: 1.0e-4
    orange: 1.0e-3
    red: 1.0e-2
  miss_distance_threshold_km: 1.0

sensors:
  lidar:
    enabled: true
    range_km: 50.0
    angular_resolution_deg: 0.1
    noise_sigma: 0.001
  radar:
    enabled: true
    range_km: 500.0
    frequency_ghz: 9.5
    range_resolution_m: 10.0
    doppler_resolution_ms: 0.1
  optical:
    enabled: true
    fov_deg: 5.0
    focal_length_mm: 2000.0
    pixel_size_um: 5.4
    snr_threshold: 3.0

space_weather:
  enabled: true
  kp_index: 3.5
  solar_flux_f107: 150.0
  geomagnetic_storm_probability: 0.05

training:
  detector:
    epochs: 100
    batch_size: 16
    lr: 0.001
    lr_scheduler: "cosine"
    warmup_epochs: 5
    weight_decay: 0.0005
  predictor:
    epochs: 200
    batch_size: 32
    lr: 0.0001
    gradient_clip: 1.0

dashboard:
  host: "0.0.0.0"
  port: 8050
  update_interval_ms: 500
  max_track_history: 200
  theme: "dark"
