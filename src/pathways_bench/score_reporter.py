from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

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
        connected_pairs_col: str = 'connected_pairs',
        na_sentinel_num: float = -99.99,
        use_pygeos: bool = False,
    ) -> None:
        if not use_pygeos:
            os.environ['USE_PYGEOS'] = '0'

        self.metrics = MetricsHelper()
        self.pred_path = Path(pred_path)
        self.gt_path = Path(gt_path) if gt_path is not None else None
        self.connected_pairs_col = connected_pairs_col
        self.na_sentinel_num = na_sentinel_num

        self.metrics.connected_pairs_col = self.connected_pairs_col

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
        mode = 'edge' if has_pairs else 'node/generic'

        # Always compute aggregate P/R/F1 if tp, fp, fn present
        precision, recall, f1 = self.metrics.compute_aggregate_f1(self.pred_gdf)

        status: Dict[str, object] = {
            'file': gname,
            'mode': mode,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

        # Traversability similarity only when we have connected_pairs in both
        if has_pairs and self.gt_gdf is not None and self.metrics.connected_pairs_col in self.gt_gdf.columns:
            ts = self.metrics.compute_tra_jaccard(self.pred_gdf, self.gt_gdf)
            status['traversability_similarity'] = ts

        return status

    def _row_ts(self, row, gt_gdf: gpd.GeoDataFrame) -> float:
        """
        Per-row Jaccard between connected-pair strings, matching original logic.
        """
        idx = row.name
        if idx not in gt_gdf.index:
            return 0.0

        try:
            pred_s = row[self.connected_pairs_col]
            gt_s = gt_gdf.loc[idx, self.connected_pairs_col]
        except Exception:
            return 0.0

        # reuse helper’s robust parser
        s1 = self.metrics._str_to_tuple_set(pred_s)
        s2 = self.metrics._str_to_tuple_set(gt_s)
        if not s1 and not s2:
            return 0.0
        union = s1 | s2
        return (len(s1 & s2) / len(union)) if union else 0.0

    def save_scores_geojson(self, out_path: str | os.PathLike | None = None) -> Optional[str]:
        """
        Adds a 'ts' column (per-TIP traversability similarity) to the predicted stats
        and saves to disk, mirroring the original '*_scores.geojson' output.
        Returns the written path, or None if not applicable.
        """
        if self.gt_gdf is None or self.connected_pairs_col not in self.pred_gdf.columns:
            return None

        # compute per-row TS
        self.pred_gdf = self.pred_gdf.copy()
        self.pred_gdf['ts'] = self.pred_gdf.apply(self._row_ts, axis=1, args=(self.gt_gdf,))

        # default output path next to the pred stats file
        if out_path is None:
            name = self.pred_path.name
            if name.endswith('stats.geojson'):
                out_path = self.pred_path.with_name(name.replace('stats.geojson', 'scores.geojson'))
            else:
                out_path = self.pred_path.with_name(self.pred_path.stem + "_scores.geojson")

        self.pred_gdf.to_file(str(out_path), driver='GeoJSON')
        return str(out_path)

