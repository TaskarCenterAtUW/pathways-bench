# helper.py
from __future__ import annotations

import os
from typing import Iterable, Tuple, Any

import geopandas as gpd
import numpy as np

CONNECTED_PAIRS_COL = "connected_pairs"


class MetricsHelper:
    """
    Small utility for computing summary metrics and handling string-encoded pairs.

    Parameters
    ----------
    connected_pairs_col : str
        Column name used for connected edge pairs.
    na_sentinel_str : str
        Sentinel string used to mark invalid connected_pairs rows.
    """

    def __init__(self, *, connected_pairs_col: str = CONNECTED_PAIRS_COL, na_sentinel_str: str = "-99.99"):
        self.connected_pairs_col = connected_pairs_col
        self.na_sentinel_str = na_sentinel_str

    # ---------- Public API ----------

    @staticmethod
    def check_file_exists(filepath: str) -> bool:
        """Raise FileNotFoundError if file is missing; return True otherwise."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No such file: {filepath}")
        return True

    def compute_aggregate_f1(self, gdf: gpd.GeoDataFrame) -> Tuple[float, float, float]:
        """
        Aggregate precision/recall/F1 using sums of tp, fp, fn.
        Returns (precision, recall, f1) rounded to 3 decimals.
        """
        tp = float(np.nansum(np.asarray(gdf.get("tp", 0))))
        fp = float(np.nansum(np.asarray(gdf.get("fp", 0))))
        fn = float(np.nansum(np.asarray(gdf.get("fn", 0))))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return round(precision, 3), round(recall, 3), round(f1, 3)

    def compute_tra_jaccard(self, gdf_pred: gpd.GeoDataFrame, gdf_gt: gpd.GeoDataFrame) -> float:
        """
        Mean Jaccard similarity over per-tile connected edge-pair sets.
        Both columns should be strings like: '(0,0) (0,2) (2,2)'.
        """
        col = self.connected_pairs_col
        pred_col = gdf_pred[col].tolist()
        gt_col = gdf_gt[col].tolist()
        pred_col, gt_col = self._remove_na_string_pairs(pred_col, gt_col)

        ious: list[float] = []
        for s1, s2 in zip(pred_col, gt_col):
            set1, set2 = self._str_to_tuple_set(s1), self._str_to_tuple_set(s2)
            if not set1 and not set2:
                continue
            union = set1 | set2
            ious.append(len(set1 & set2) / len(union) if union else 0.0)

        return float(np.mean(ious)) if ious else 0.0

    # ---------- Private helpers ----------

    def _remove_na_string_pairs(self, s1: Iterable[str], s2: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
        """Filter out rows where either string equals the NA sentinel string."""
        a1 = np.asarray(list(s1), dtype=object)
        a2 = np.asarray(list(s2), dtype=object)
        m = (a1 != self.na_sentinel_str) & (a2 != self.na_sentinel_str)
        return a1[m], a2[m]

    @staticmethod
    def _str_to_tuple_set(data_str: str) -> set[Any] | set[tuple[int, ...]]:
        """'(0,0) (1,2)' -> {(0,0), (1,2)}; empty set on malformed input."""
        if not isinstance(data_str, str) or not data_str.strip():
            return set()
        try:
            items = data_str.replace("(", "").replace(")", "").split()
            return {tuple(map(int, t.split(","))) for t in items if "," in t}
        except Exception:
            return set()
