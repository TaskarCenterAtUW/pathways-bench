from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import geopandas as gpd
from .helpers import MetricsHelper


class ScoreReporter:
    """
    Compute summary metrics for edge/node stats files.

    Parameters
    ----------
    pred_path : str | os.PathLike
        Path to the *predicted* stats GeoJSON/GeoPackage/etc.
    gt_path : str | os.PathLike | None
        Path to the *ground truth* stats file. Required if you want
        traversability Jaccard (i.e., connected_pairs comparison).
    connected_pairs_col : str
        Column containing connected edge-pair strings (default: 'connected_pairs').
    na_sentinel_num : float
        Numeric sentinel value used to mark invalid rows (default: -99.99).
    use_pygeos : bool
        If False, sets USE_PYGEOS=0 for Shapely 2 compatibility environments.
    """

    def __init__(
        self,
        pred_path: str | os.PathLike,
        gt_path: str | os.PathLike | None = None,
        *,
        connected_pairs_col: str = "connected_pairs",
        na_sentinel_num: float = -99.99,
        use_pygeos: bool = False,
    ) -> None:
        if not use_pygeos:
            os.environ["USE_PYGEOS"] = "0"

        self.metrics = MetricsHelper()
        self.pred_path = Path(pred_path)
        self.gt_path = Path(gt_path) if gt_path is not None else None
        self.connected_pairs_col = connected_pairs_col
        self.na_sentinel_num = na_sentinel_num

        # Load files
        self.pred_gdf = gpd.read_file(self.pred_path)
        self.gt_gdf = gpd.read_file(self.gt_path) if self.gt_path else None

    # -------------------------
    # Public API
    # -------------------------

    def run(self) -> Dict[str, object]:
        """
        Compute and return a status dict with summary metrics.

        Returns
        -------
        dict
            Keys:
              - file: str (pred file name)
              - mode: 'edge' if connected_pairs present; else 'node/generic'
              - precision, recall, f1: floats (rounded to 3 decimals)
              - traversability_similarity: float (if connected_pairs present and gt provided)
        """
        gname = self.pred_path.name
        has_pairs = self.connected_pairs_col in self.pred_gdf.columns
        mode = "edge" if has_pairs else "node/generic"

        # Always compute aggregate P/R/F1 if tp, fp, fn present
        precision, recall, f1 = self.metrics.compute_aggregate_f1(self.pred_gdf)

        status: Dict[str, object] = {
            "file": gname,
            "mode": mode,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        # Traversability similarity only when we have connected_pairs in both
        if has_pairs and self.gt_gdf is not None and self.metrics.connected_pairs_col in self.gt_gdf.columns:
            ts = self.metrics.compute_tra_jaccard(self.pred_gdf, self.gt_gdf)
            status["traversability_similarity"] = ts

        return status

