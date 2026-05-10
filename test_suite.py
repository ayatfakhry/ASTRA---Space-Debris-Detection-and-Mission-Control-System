"""
metrics.py - ASTRA Performance Evaluation Suite
Detection (mAP), Tracking (MOTA/IDF1), Prediction (RMSE/FDE), and
Collision (F1/ROC) metrics per COCO and MOT17 benchmark standards.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


# ── Detection Metrics ──────────────────────────────────────────────────────────

def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    xi1 = max(box_a[0], box_b[0]); yi1 = max(box_a[1], box_b[1])
    xi2 = min(box_a[2], box_b[2]); yi2 = min(box_a[3], box_b[3])
    inter = max(0.0, xi2-xi1) * max(0.0, yi2-yi1)
    a_a   = (box_a[2]-box_a[0])*(box_a[3]-box_a[1])
    a_b   = (box_b[2]-box_b[0])*(box_b[3]-box_b[1])
    union = a_a + a_b - inter + 1e-9
    return inter / union


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute Average Precision using 101-point interpolation (COCO standard)."""
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        p = precisions[recalls >= t]
        ap += (np.max(p) if len(p) > 0 else 0.0)
    return ap / 101


def compute_map(predictions: List[dict], ground_truths: List[dict],
                iou_thresholds: List[float] = None,
                n_classes: int = 4) -> dict:
    """
    Compute mAP@0.5 and mAP@0.5:0.95 for multi-class detection.

    predictions: [{image_id, class_id, bbox, score}]
    ground_truths: [{image_id, class_id, bbox}]
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5] + list(np.arange(0.5, 1.0, 0.05))

    ap_results = {}
    for cls in range(n_classes):
        cls_preds = [p for p in predictions  if p["class_id"]==cls]
        cls_gts   = [g for g in ground_truths if g["class_id"]==cls]

        ap_per_iou = []
        for iou_t in iou_thresholds:
            ap = _compute_class_ap(cls_preds, cls_gts, iou_t)
            ap_per_iou.append(ap)
        ap_results[cls] = {"ap50": ap_per_iou[0],
                            "ap":  float(np.mean(ap_per_iou))}

    map50  = float(np.mean([v["ap50"] for v in ap_results.values()]))
    map5095= float(np.mean([v["ap"]   for v in ap_results.values()]))

    return {
        "mAP@0.5":      round(map50, 4),
        "mAP@0.5:0.95": round(map5095, 4),
        "per_class":    {k: {kk: round(vv,4) for kk,vv in v.items()}
                         for k, v in ap_results.items()},
    }


def _compute_class_ap(preds, gts, iou_t) -> float:
    if not preds or not gts:
        return 0.0
    preds = sorted(preds, key=lambda x: -x["score"])
    gt_by_img = {}
    for g in gts:
        gt_by_img.setdefault(g["image_id"],[]).append(g)
    matched  = set()
    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))
    for i, p in enumerate(preds):
        img_gts = gt_by_img.get(p["image_id"], [])
        best_iou= 0.0; best_j = -1
        for j, g in enumerate(img_gts):
            iou = compute_iou(np.array(p["bbox"]), np.array(g["bbox"]))
            if iou > best_iou:
                best_iou = iou; best_j = j
        key = (p["image_id"], best_j)
        if best_iou >= iou_t and key not in matched:
            tp[i] = 1.0; matched.add(key)
        else:
            fp[i] = 1.0
    tp_cum = np.cumsum(tp); fp_cum = np.cumsum(fp)
    recall    = tp_cum / max(len(gts), 1)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    return compute_ap(recall, precision)


# ── Tracking Metrics ───────────────────────────────────────────────────────────

@dataclass
class MOTAccumulator:
    """
    MOT challenge accumulator for MOTA, MOTP, IDF1 computation.
    Implements Bernardin & Stiefelhagen (2008) MOTA metric.
    """
    tp:  int = 0
    fp:  int = 0
    fn:  int = 0
    id_switches: int = 0
    sum_dist_m:  float = 0.0
    n_matches:   int   = 0
    _prev_matches: dict = field(default_factory=dict)

    def update(self, gt_tracks: List[dict], pred_tracks: List[dict],
               dist_thresh_m: float = 2000.0):
        """Update accumulator with one frame's assignments."""
        if not gt_tracks and not pred_tracks:
            return
        # Hungarian matching
        if gt_tracks and pred_tracks:
            cost = np.zeros((len(gt_tracks), len(pred_tracks)))
            for i, gt in enumerate(gt_tracks):
                for j, pr in enumerate(pred_tracks):
                    gp = np.array(gt.get("pos_km",[0,0,0]))*1e3
                    pp = np.array(pr.get("pos_km",[0,0,0]))*1e3
                    cost[i,j] = np.linalg.norm(gp-pp)
            from scipy.optimize import linear_sum_assignment
            ri, ci = linear_sum_assignment(cost)
            matched_gt, matched_pr = set(), set()
            curr_matches = {}
            for r, c in zip(ri, ci):
                if cost[r,c] < dist_thresh_m:
                    self.tp += 1
                    self.sum_dist_m += cost[r,c]
                    self.n_matches  += 1
                    gt_id = gt_tracks[r].get("id",r)
                    pr_id = pred_tracks[c].get("id",c)
                    if self._prev_matches.get(gt_id) not in (None, pr_id):
                        self.id_switches += 1
                    curr_matches[gt_id] = pr_id
                    matched_gt.add(r); matched_pr.add(c)
            self.fn += len(gt_tracks)  - len(matched_gt)
            self.fp += len(pred_tracks)- len(matched_pr)
            self._prev_matches = curr_matches
        else:
            self.fn += len(gt_tracks)
            self.fp += len(pred_tracks)

    def compute(self) -> dict:
        n_gt   = self.tp + self.fn
        mota   = 1.0 - (self.fp+self.fn+self.id_switches) / max(n_gt,1)
        motp   = self.sum_dist_m / max(self.n_matches, 1)
        recall = self.tp / max(n_gt, 1)
        prec   = self.tp / max(self.tp+self.fp, 1)
        idf1   = 2*self.tp / max(2*self.tp+self.fp+self.fn, 1)
        return {
            "MOTA":       round(mota, 4),
            "MOTP_m":     round(motp, 2),
            "IDF1":       round(idf1, 4),
            "Recall":     round(recall, 4),
            "Precision":  round(prec, 4),
            "ID_Switches":self.id_switches,
            "TP": self.tp, "FP": self.fp, "FN": self.fn,
        }


# ── Trajectory Prediction Metrics ─────────────────────────────────────────────

def trajectory_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """
    Compute trajectory prediction errors.

    pred: (B, T, 3) predicted positions [km]
    gt:   (B, T, 3) ground-truth positions [km]
    """
    diff   = pred - gt                     # (B, T, 3)
    dist   = np.linalg.norm(diff, axis=-1) # (B, T)
    ade    = float(dist.mean())            # Average Displacement Error
    fde    = float(dist[:,-1].mean())      # Final Displacement Error
    rmse   = float(np.sqrt((diff**2).mean()))
    # Per-timestep RMSE
    rmse_t = np.sqrt((diff**2).mean(axis=(0,2)))  # (T,)

    return {
        "ADE_km":     round(ade,  4),
        "FDE_km":     round(fde,  4),
        "RMSE_km":    round(rmse, 4),
        "RMSE_t30s":  round(float(rmse_t[min(29,len(rmse_t)-1)]),4),
        "RMSE_t60s":  round(float(rmse_t[min(59,len(rmse_t)-1)]),4),
        "max_err_km": round(float(dist.max()), 4),
    }


# ── Collision Prediction Metrics ───────────────────────────────────────────────

def collision_prediction_metrics(
        predicted_pcs: np.ndarray,
        true_collisions: np.ndarray,
        threshold: float = 1e-4) -> dict:
    """
    Binary classification metrics for collision prediction.
    predicted_pcs:  array of predicted Pc values
    true_collisions: binary array (1=collision occurred)
    """
    predicted = (predicted_pcs >= threshold).astype(int)
    tp = int(np.sum((predicted==1) & (true_collisions==1)))
    fp = int(np.sum((predicted==1) & (true_collisions==0)))
    fn = int(np.sum((predicted==0) & (true_collisions==1)))
    tn = int(np.sum((predicted==0) & (true_collisions==0)))
    prec = tp / max(tp+fp, 1)
    rec  = tp / max(tp+fn, 1)
    f1   = 2*prec*rec / max(prec+rec, 1e-9)
    acc  = (tp+tn) / max(tp+fp+fn+tn, 1)
    return {
        "Precision":  round(prec, 4),
        "Recall":     round(rec,  4),
        "F1":         round(f1,   4),
        "Accuracy":   round(acc,  4),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


# ── System Benchmark ───────────────────────────────────────────────────────────

class SystemBenchmark:
    """End-to-end system performance benchmarking."""

    def __init__(self):
        self.results = {}

    def run_detection_benchmark(self, n_images=1000) -> dict:
        """Simulate detection benchmark with synthetic data."""
        rng   = np.random.default_rng(0)
        preds, gts = [], []
        for img_id in range(n_images):
            n_gt  = rng.integers(1,8)
            for j in range(n_gt):
                x1,y1 = rng.uniform(0,0.8,2)
                gts.append({"image_id":img_id,"class_id":int(rng.integers(0,4)),
                             "bbox":[x1,y1,x1+0.1,y1+0.1]})
                # TP pred with 85% chance
                if rng.random()<0.85:
                    noise = rng.normal(0,0.02,4)
                    preds.append({"image_id":img_id,"class_id":int(rng.integers(0,4)),
                                  "bbox":[x1+noise[0],y1+noise[1],
                                          x1+0.1+noise[2],y1+0.1+noise[3]],
                                  "score":rng.uniform(0.5,1.0)})
        result = compute_map(preds, gts)
        self.results["detection"] = result
        return result

    def run_tracking_benchmark(self, n_frames=500) -> dict:
        """Simulate tracking benchmark."""
        acc = MOTAccumulator()
        rng = np.random.default_rng(0)
        for _ in range(n_frames):
            n = rng.integers(2,10)
            gts  = [{"id":i,"pos_km":[rng.uniform(-100,100) for _ in range(3)]}
                    for i in range(n)]
            preds= [{"id":i,"pos_km":[g["pos_km"][k]+rng.normal(0,0.5)
                    for k in range(3)]} for i,g in enumerate(gts)
                    if rng.random()>0.1]
            acc.update(gts, preds)
        result = acc.compute()
        self.results["tracking"] = result
        return result

    def run_prediction_benchmark(self, n_samples=1000, T=60) -> dict:
        """Simulate trajectory prediction benchmark."""
        rng  = np.random.default_rng(0)
        gt   = rng.normal(0, 100, (n_samples, T, 3))
        pred = gt + rng.normal(0, 0.1*(1+np.arange(T)[None,:,None]/T), gt.shape)
        result = trajectory_metrics(pred/1000, gt/1000)  # convert to km
        self.results["prediction"] = result
        return result

    def print_report(self):
        """Print formatted benchmark report."""
        print("\n" + "="*64)
        print("  ASTRA SYSTEM PERFORMANCE BENCHMARK REPORT")
        print("="*64)
        for module, res in self.results.items():
            print(f"\n  [{module.upper()}]")
            for k,v in res.items():
                print(f"    {k:<25s}: {v}")
        print("="*64 + "\n")


if __name__ == "__main__":
    bench = SystemBenchmark()
    bench.run_detection_benchmark()
    bench.run_tracking_benchmark()
    bench.run_prediction_benchmark()
    bench.print_report()
