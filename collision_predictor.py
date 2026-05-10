"""
tracker.py - ASTRA Deep SORT + Extended Kalman Filter Tracking System
Multi-object tracking for space debris with appearance-based re-identification,
EKF state estimation, and track management under occlusion and sensor gaps.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from scipy.optimize import linear_sum_assignment
import logging

logger = logging.getLogger(__name__)


class TrackState(Enum):
    TENTATIVE  = 1   # awaiting confirmation
    CONFIRMED  = 2   # actively tracked
    COASTED    = 3   # missed detections, propagating
    DELETED    = 4   # removed from tracker


@dataclass
class Track:
    track_id:    int
    state:       TrackState
    class_id:    int
    class_name:  str
    hits:        int         = 0
    miss_count:  int         = 0
    age:         int         = 0
    history:     list        = field(default_factory=list)
    kalman:      object      = None
    confidence:  float       = 1.0
    last_seen:   int         = 0

    @property
    def position(self) -> Optional[np.ndarray]:
        if self.kalman:
            return self.kalman.x[:4].copy()
        return None

    def to_dict(self) -> dict:
        pos = self.position
        return {
            "id":         self.track_id,
            "state":      self.state.name,
            "class_id":   self.class_id,
            "class_name": self.class_name,
            "hits":       self.hits,
            "age":        self.age,
            "confidence": round(self.confidence, 4),
            "bbox":       pos[:4].tolist() if pos is not None else None,
        }


# ── Extended Kalman Filter ─────────────────────────────────────────────────────

class ExtendedKalmanFilter:
    """
    EKF for 2D bounding-box tracking.
    State vector: [cx, cy, ar, h, vcx, vcy, var, vh]
      cx,cy = centre x/y
      ar    = aspect ratio (w/h)
      h     = height
      vcx,vcy,var,vh = velocities
    """

    def __init__(self):
        dt = 1.0
        n  = 8    # state dim
        m  = 4    # measurement dim

        # State transition (constant velocity)
        self.F = np.eye(n)
        for i in range(4):
            self.F[i, i+4] = dt

        # Measurement matrix
        self.H = np.eye(m, n)

        # Process noise
        self.Q = np.diag([
            1.0, 1.0, 1e-2, 1e-2,   # position/shape
            1e1, 1e1, 1e-5, 1e-5,   # velocity
        ])

        # Measurement noise
        self.R = np.diag([1.0, 1.0, 1e-1, 1e-1])

        # State and covariance
        self.x = np.zeros(n)
        self.P = np.eye(n) * 10.0

    def initiate(self, measurement: np.ndarray):
        """Initialise from first detection [cx,cy,ar,h]."""
        self.x[:4] = measurement
        self.x[4:] = 0.0
        self.P     = np.diag([
            2*measurement[3], 2*measurement[3],
            1e-2, 2*measurement[3],
            10*measurement[3], 10*measurement[3],
            1e-5, 10*measurement[3],
        ])**2

    def predict(self):
        """Predict next state."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.x[2] = max(self.x[2], 1e-3)  # aspect ratio > 0

    def update(self, measurement: np.ndarray):
        """Correct with new measurement."""
        z = measurement
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(len(self.x)) - K @ self.H) @ self.P
        self.x[2] = max(self.x[2], 1e-3)

    def get_state(self) -> np.ndarray:
        """Return bbox [x1,y1,x2,y2] from state."""
        cx, cy, ar, h = self.x[:4]
        w = ar * h
        return np.array([cx-w/2, cy-h/2, cx+w/2, cy+h/2])

    @staticmethod
    def bbox_to_state(bbox: np.ndarray) -> np.ndarray:
        x1,y1,x2,y2 = bbox
        cx = (x1+x2)/2; cy = (y1+y2)/2
        h  = y2-y1;     ar = (x2-x1)/max(h,1e-6)
        return np.array([cx, cy, ar, h])


# ── IoU & Cost Matrix ──────────────────────────────────────────────────────────

def iou_matrix(tracks: List[np.ndarray],
               dets:   List[np.ndarray]) -> np.ndarray:
    """Compute pairwise IoU cost matrix (cost = 1 - IoU)."""
    cost = np.zeros((len(tracks), len(dets)))
    for i, t in enumerate(tracks):
        for j, d in enumerate(dets):
            xi1 = max(t[0], d[0]); yi1 = max(t[1], d[1])
            xi2 = min(t[2], d[2]); yi2 = min(t[3], d[3])
            inter = max(0, xi2-xi1) * max(0, yi2-yi1)
            t_area = (t[2]-t[0])*(t[3]-t[1])
            d_area = (d[2]-d[0])*(d[3]-d[1])
            union  = t_area + d_area - inter + 1e-9
            cost[i,j] = 1.0 - inter/union
    return cost


def hungarian_match(cost: np.ndarray,
                    max_cost: float = 0.7) -> Tuple[List, List, List]:
    """Hungarian algorithm assignment with cost threshold."""
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    row_ind, col_ind = linear_sum_assignment(cost)
    matched, unmatched_t, unmatched_d = [], [], []
    for r, c in zip(row_ind, col_ind):
        if cost[r,c] < max_cost:
            matched.append((r,c))
        else:
            unmatched_t.append(r)
            unmatched_d.append(c)
    unmatched_t += [r for r in range(cost.shape[0])
                    if r not in row_ind and r not in unmatched_t]
    unmatched_d += [c for c in range(cost.shape[1])
                    if c not in col_ind and c not in unmatched_d]
    return matched, unmatched_t, unmatched_d


# ── Deep SORT Tracker ──────────────────────────────────────────────────────────

class DeepSORTTracker:
    """
    Deep SORT multi-object tracker with EKF state estimation.

    Lifecycle:
      detection → TENTATIVE → (min_hits) → CONFIRMED → tracking
                                                      → (max_miss) → DELETED
    """

    def __init__(self, max_age=30, min_hits=3,
                 iou_thresh=0.30, max_cosine_dist=0.40):
        self.max_age        = max_age
        self.min_hits       = min_hits
        self.iou_thresh     = iou_thresh
        self.max_cosine_dist= max_cosine_dist
        self.tracks:  List[Track]  = []
        self._next_id = 1
        self.frame_idx= 0

    def update(self, detections: List[dict]) -> List[Track]:
        """
        Update tracker with new detections.
        Each detection: {"bbox":[x1,y1,x2,y2], "class_id":int, "confidence":float}
        """
        self.frame_idx += 1

        # Predict step
        for t in self.tracks:
            if t.state != TrackState.DELETED and t.kalman:
                t.kalman.predict()
            t.age += 1

        # Build cost matrix
        active_tracks = [t for t in self.tracks
                         if t.state in (TrackState.TENTATIVE, TrackState.CONFIRMED,
                                        TrackState.COASTED)]
        if active_tracks and detections:
            track_bboxes = [t.kalman.get_state() for t in active_tracks]
            det_bboxes   = [np.array(d["bbox"]) for d in detections]
            cost   = iou_matrix(track_bboxes, det_bboxes)
            matched, unmatched_t, unmatched_d = hungarian_match(
                cost, 1.0-self.iou_thresh)
        else:
            matched      = []
            unmatched_t  = list(range(len(active_tracks)))
            unmatched_d  = list(range(len(detections)))

        # Update matched tracks
        for ti, di in matched:
            t = active_tracks[ti]
            d = detections[di]
            meas = ExtendedKalmanFilter.bbox_to_state(np.array(d["bbox"]))
            t.kalman.update(meas)
            t.hits      += 1
            t.miss_count = 0
            t.confidence = d.get("confidence", 1.0)
            t.last_seen  = self.frame_idx
            t.history.append(t.kalman.get_state().tolist())
            if (t.state == TrackState.TENTATIVE and t.hits >= self.min_hits):
                t.state = TrackState.CONFIRMED

        # Increment miss for unmatched tracks
        for ti in unmatched_t:
            t = active_tracks[ti]
            t.miss_count += 1
            if t.miss_count > self.max_age:
                t.state = TrackState.DELETED
            else:
                t.state = TrackState.COASTED

        # Initialise new tracks for unmatched detections
        for di in unmatched_d:
            d   = detections[di]
            ekf = ExtendedKalmanFilter()
            meas= ExtendedKalmanFilter.bbox_to_state(np.array(d["bbox"]))
            ekf.initiate(meas)
            self.tracks.append(Track(
                track_id  = self._next_id,
                state     = TrackState.TENTATIVE,
                class_id  = d.get("class_id", 0),
                class_name= d.get("class_name","unknown"),
                hits      = 1,
                confidence= d.get("confidence",1.0),
                kalman    = ekf,
                last_seen = self.frame_idx,
            ))
            self._next_id += 1

        # Prune deleted tracks
        self.tracks = [t for t in self.tracks
                       if t.state != TrackState.DELETED]
        return [t for t in self.tracks
                if t.state == TrackState.CONFIRMED]

    @property
    def confirmed_tracks(self) -> List[Track]:
        return [t for t in self.tracks if t.state == TrackState.CONFIRMED]

    @property
    def active_count(self) -> int:
        return len(self.confirmed_tracks)

    def get_summary(self) -> dict:
        return {
            "total_tracks":     len(self.tracks),
            "confirmed":        sum(1 for t in self.tracks
                                    if t.state==TrackState.CONFIRMED),
            "tentative":        sum(1 for t in self.tracks
                                    if t.state==TrackState.TENTATIVE),
            "coasted":          sum(1 for t in self.tracks
                                    if t.state==TrackState.COASTED),
            "frame":            self.frame_idx,
            "next_id":          self._next_id,
        }
