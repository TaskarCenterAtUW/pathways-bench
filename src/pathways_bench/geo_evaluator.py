from __future__ import annotations
import os
import itertools
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import dask_geopandas
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely import LineString, MultiLineString, MultiPolygon, Point, Polygon

# helpers
from .helpers import MetricsHelper


GeoFrameOrPath = Union[str, os.PathLike, gpd.GeoDataFrame]


class GeoStatsEvaluator:
    """
    Evaluate predicted edges/nodes against ground truth over tiled areas.

    Parameters
    ----------
    proj : str
        Target CRS for all inputs (default: 'epsg:26910').
    precision : float
        Tolerance used when snapping nodes to polygon edges (default: 1e-5).
    buffer_size : float
        Buffer distance (in CRS units) for line matching (default: 5).
    e_threshold : float
        Error threshold (in CRS units) for match acceptance (default: 5).
    num_partitions : int
        Dask partitions for tile parallelism (default: 32).
    curb_pred_filter : tuple[str, str]
        Column/value selecting curb nodes in predictions (default: ('ext:node_type','curb')).
    curb_gt_filter : tuple[str, str]
        Column/value selecting curb nodes in ground truth (default: ('barrier','kerb')).
    """

    def __init__(
        self,
        *,
        proj: str = 'epsg:26910',
        precision: float = 1e-5,
        buffer_size: float = 5.0,
        e_threshold: float = 5.0,
        num_partitions: int = 32,
        curb_pred_filter: Tuple[str, str] = ("ext:node_type", "curb"),
        curb_gt_filter: Tuple[str, str] = ("barrier", "kerb"),
        output: str | os.PathLike | None = None,
        use_pygeos: bool = False,
    ) -> None:
        if not use_pygeos:
            os.environ["USE_PYGEOS"] = "0"
        self.proj = proj
        self.precision = precision
        self.buffer_size = buffer_size
        self.e_threshold = e_threshold
        self.num_partitions = num_partitions
        self.curb_pred_filter = curb_pred_filter
        self.curb_gt_filter = curb_gt_filter
        self.output_dir = Path(output) if output else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.helper = MetricsHelper()

    # =========================
    # Public, parameter-based API
    # =========================

    def run(
        self,
        *,
        tile: GeoFrameOrPath,
        edges: GeoFrameOrPath,
        gt_edges: GeoFrameOrPath,
        nodes: GeoFrameOrPath | None = None,
        gt_nodes: GeoFrameOrPath | None = None,
        save_paths: Dict[str, str] | None = None,
    ) -> Dict[str, object]:
        """
        High-level orchestration. Accepts paths or GeoDataFrames.
        Returns a dict with GeoDataFrames and summary numbers. Optionally writes out files.

        Keys returned (when present):
            - pred_edge_stats, gt_edge_stats, edge_summary
            - curb_stats, curb_link_stats, curb_summary, curb_link_summary
        """

        if save_paths is None:
            save_paths = {}
            # If output_dir exists, put everything there
            if self.output_dir:
                base_pred = Path(edges).stem if isinstance(edges, (str, os.PathLike)) else "pred"
                base_gt = Path(gt_edges).stem if isinstance(gt_edges, (str, os.PathLike)) else "gt"
                save_paths["pred_edge_stats"] = str(self.output_dir / f"{base_pred}_stats.geojson")
                save_paths["gt_edge_stats"] = str(self.output_dir / f"{base_gt}_gt_stats.geojson")
                if nodes is not None:
                    base_nodes = Path(nodes).stem if isinstance(nodes, (str, os.PathLike)) else "curbs"
                    save_paths["curb_stats"] = str(self.output_dir / f"{base_nodes}_curb_stats.geojson")
                    save_paths["curb_link_stats"] = str(self.output_dir / f"{base_nodes}_curb_link_stats.geojson")
            else:
                # Fall back to original-file-based paths
                if isinstance(edges, (str, os.PathLike)):
                    base = Path(edges)
                    save_paths["pred_edge_stats"] = str(base.with_name(f"{base.stem}_stats.geojson"))
                if isinstance(gt_edges, (str, os.PathLike)):
                    base = Path(gt_edges)
                    save_paths["gt_edge_stats"] = str(base.with_name(f"{base.stem}_gt_stats.geojson"))
                if nodes is not None and isinstance(nodes, (str, os.PathLike)):
                    base = Path(nodes)
                    save_paths["curb_stats"] = str(base.with_name(f"{base.stem}_curb_stats.geojson"))
                    save_paths["curb_link_stats"] = str(base.with_name(f"{base.stem}_curb_link_stats.geojson"))

        tile_gdf = self._coerce_gdf(tile)
        edges_gdf = self._coerce_gdf(edges)
        gt_edges_gdf = self._coerce_gdf(gt_edges)

        # --- EDGE EVAL ---
        pred_edge_stats = self.evaluate_edges(tile_gdf, edges_gdf, gt_edges_gdf)
        gt_edge_stats = self.evaluate_edges(tile_gdf, gt_edges_gdf, gt_edges_gdf)

        saved_paths: Dict[str, str] = {}
        if path := save_paths.get("pred_edge_stats"):
            pred_edge_stats.to_file(path, driver="GeoJSON")
            saved_paths["pred_edge_stats"] = path
        if path := save_paths.get("gt_edge_stats"):
            gt_edge_stats.to_file(path, driver="GeoJSON")
            saved_paths["gt_edge_stats"] = path

        tra, p, r, f1 = self.summarise_edge_stats(pred_edge_stats, gt_edge_stats)

        out: Dict[str, object] = {
            "pred_edge_stats": pred_edge_stats,
            "gt_edge_stats": gt_edge_stats,
            "edge_summary": {
                "traversability": tra,
                "precision": p,
                "recall": r,
                "f1": f1,
            },
        }

        # Optional nodes (curbs + curb-links)
        if nodes is not None and gt_nodes is not None:
            nodes_gdf = self._coerce_gdf(nodes)
            gt_nodes_gdf = self._coerce_gdf(gt_nodes)

            curb_stats, curb_link_stats = self.evaluate_curbs_and_links(
                tile_gdf, nodes_gdf, gt_nodes_gdf, edges_gdf, gt_edges_gdf
            )

            # Save node stats if paths provided
            if path := save_paths.get("curb_stats"):
                curb_stats.to_file(path, driver="GeoJSON")
                saved_paths["curb_stats"] = path
            if path := save_paths.get("curb_link_stats"):
                curb_link_stats.to_file(path, driver="GeoJSON")
                saved_paths["curb_link_stats"] = path

            out.update(
                {
                    "curb_stats": curb_stats,
                    "curb_link_stats": curb_link_stats,
                    "curb_summary": self._summary_dict(*self.summarise_node_stats(curb_stats)),
                    "curb_link_summary": self._summary_dict(
                        *self.summarise_node_stats(curb_link_stats)
                    ),
                }
            )

        # Include saved paths (only if something was saved)
        if saved_paths:
            out["saved_paths"] = saved_paths

        return out

    def evaluate_edges(
        self,
        tile_gdf: GeoFrameOrPath,
        edges_gdf: GeoFrameOrPath,
        gt_edges_gdf: GeoFrameOrPath,
    ) -> gpd.GeoDataFrame:
        """Per-tile edge metrics. Accepts paths or GeoDataFrames."""
        tile_gdf = self._coerce_gdf(tile_gdf)
        edges_gdf = self._coerce_gdf(edges_gdf)
        gt_edges_gdf = self._coerce_gdf(gt_edges_gdf)

        df_dask = dask_geopandas.from_geopandas(tile_gdf, npartitions=self.num_partitions)
        meta_edges = gpd.GeoDataFrame(
            {
                "total_edges": pd.Series(dtype="float64"),
                "connect_edges": pd.Series(dtype="float64"),
                "connected_pairs": pd.Series(dtype="object"),
                "tp": pd.Series(dtype="float64"),
                "fp": pd.Series(dtype="float64"),
                "fn": pd.Series(dtype="float64"),
            },
            geometry=gpd.GeoSeries([], dtype="geometry"),
            crs=tile_gdf.crs,
        )
        return df_dask.apply(
            self._compute_edge_score,
            axis=1,
            meta=meta_edges,
            gdf=edges_gdf,
            gdf_gt=gt_edges_gdf,
        ).compute(scheduler="multiprocessing")

    def evaluate_nodes(
        self,
        tile_gdf: GeoFrameOrPath,
        nodes_gdf: GeoFrameOrPath,
        gt_nodes_gdf: GeoFrameOrPath,
    ) -> gpd.GeoDataFrame:
        """Per-tile node metrics. Accepts paths or GeoDataFrames."""
        tile_gdf = self._coerce_gdf(tile_gdf)
        nodes_gdf = self._coerce_gdf(nodes_gdf)
        gt_nodes_gdf = self._coerce_gdf(gt_nodes_gdf)

        df_dask = dask_geopandas.from_geopandas(tile_gdf, npartitions=self.num_partitions)
        meta_nodes = gpd.GeoDataFrame(
            {
                "tp": pd.Series(dtype="float64"),
                "fp": pd.Series(dtype="float64"),
                "fn": pd.Series(dtype="float64"),
            },
            geometry=gpd.GeoSeries([], dtype="geometry"),
            crs=tile_gdf.crs,
        )
        return df_dask.apply(
            self._compute_node_score,
            axis=1,
            meta=meta_nodes,
            gdf=nodes_gdf,
            gdf_gt=gt_nodes_gdf,
        ).compute(scheduler="multiprocessing")

    def evaluate_curbs_and_links(
        self,
        tile_gdf: GeoFrameOrPath,
        nodes_gdf: GeoFrameOrPath,
        gt_nodes_gdf: GeoFrameOrPath,
        edges_gdf: GeoFrameOrPath,
        gt_edges_gdf: GeoFrameOrPath,
    ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """Per-tile stats for curb nodes and curb-link nodes."""
        tile_gdf = self._coerce_gdf(tile_gdf)
        nodes_gdf = self._coerce_gdf(nodes_gdf)
        gt_nodes_gdf = self._coerce_gdf(gt_nodes_gdf)
        edges_gdf = self._coerce_gdf(edges_gdf)
        gt_edges_gdf = self._coerce_gdf(gt_edges_gdf)

        pred_col, pred_val = self.curb_pred_filter
        gt_col, gt_val = self.curb_gt_filter

        pred_curb = nodes_gdf[nodes_gdf.get(pred_col) == pred_val]
        gt_curb = gt_nodes_gdf[gt_nodes_gdf.get(gt_col) == gt_val]

        curb_stats = self.evaluate_nodes(tile_gdf, pred_curb, gt_curb)

        pred_curb_link = self._join_curb_to_edges(pred_curb, nodes_gdf, edges_gdf)
        gt_curb_link = self._join_curb_to_edges(gt_curb, gt_nodes_gdf, gt_edges_gdf)

        curb_link_stats = self.evaluate_nodes(tile_gdf, pred_curb_link, gt_curb_link)
        return curb_stats, curb_link_stats

    def summarise_edge_stats(
        self, pred_stats: gpd.GeoDataFrame, gt_stats: gpd.GeoDataFrame
    ) -> Tuple[float, float, float, float]:
        """(traversability, precision, recall, f1)"""
        tra = self.helper.compute_tra_jaccard(pred_stats, gt_stats)
        precision, recall, f1 = self.helper.compute_aggregate_f1(pred_stats)
        return tra, precision, recall, f1

    def summarise_node_stats(self, node_stats: gpd.GeoDataFrame) -> Tuple[float, float, float]:
        """(precision, recall, f1)"""
        return self.helper.compute_aggregate_f1(node_stats)

    # =========================
    # Internal helpers
    # =========================

    def _coerce_gdf(self, obj: GeoFrameOrPath) -> gpd.GeoDataFrame:
        # accept str, Path, or any os.PathLike
        if isinstance(obj, (str, os.PathLike, Path)):
            path = os.fspath(obj)  # safely convert PathLike -> str
            gdf = gpd.read_file(path)
        elif isinstance(obj, gpd.GeoDataFrame):
            gdf = obj
        else:
            raise TypeError(
                f"Expected str | os.PathLike | GeoDataFrame, got {type(obj).__name__}"
            )

        if gdf.crs is None:
            return gdf
        return gdf.to_crs(self.proj) if str(gdf.crs).lower() != self.proj.lower() else gdf

    def _add_edges_from_linestring(
        self, graph: nx.Graph, linestring: LineString, edge_attrs: Dict
    ) -> None:
        points = list(linestring.coords)
        for start, end in zip(points[:-1], points[1:]):
            graph.add_edge(start, end, **edge_attrs)

    def _graph_from_gdf(self, gdf: gpd.GeoDataFrame) -> nx.Graph:
        G = nx.Graph()
        for _, row in gdf.iterrows():
            geom = row.geometry
            attrs = row.to_dict()
            if isinstance(geom, LineString):
                self._add_edges_from_linestring(G, geom, attrs)
            elif isinstance(geom, MultiLineString):
                for ls in geom.geoms:
                    self._add_edges_from_linestring(G, ls, attrs)
        return G

    def _group_graph_points(self, G: nx.Graph, polygon: Polygon) -> Dict[int, List[Tuple[float, float]]]:
        P = polygon
        if isinstance(P, MultiPolygon) and len(P.geoms) == 1:
            P = P.geoms[0]
        boundary = list(P.boundary.coords)
        segments = [LineString([boundary[i], boundary[i + 1]]) for i in range(len(boundary) - 1)]
        segment_point_map: Dict[int, List[Tuple[float, float]]] = {i: [] for i in range(len(segments))}
        for node in G.nodes():
            pt = Point(node)
            for idx, seg in enumerate(segments):
                if seg.distance(pt) < self.precision:
                    segment_point_map[idx].append((pt.x, pt.y))
                    break
        return segment_point_map

    def _edges_are_connected(
        self,
        G: nx.Graph,
        e1_pts: Iterable[Tuple[float, float]],
        e2_pts: Iterable[Tuple[float, float]],
    ) -> bool:
        for pt1 in e1_pts:
            for pt2 in e2_pts:
                if pt1 != pt2 and nx.has_path(G, pt1, pt2):
                    return True
        return False

    def _tile_traversability_score(
        self, G: nx.Graph, polygon: Polygon
    ) -> Tuple[int, int, List[Tuple[int, int]]]:
        pts_line_map = self._group_graph_points(G, polygon)
        edge_pairs = list(itertools.combinations_with_replacement(pts_line_map.keys(), 2))
        n_total = len(edge_pairs)
        n_connected = 0
        connected_pairs: List[Tuple[int, int]] = []
        for a, b in edge_pairs:
            if self._edges_are_connected(G, pts_line_map[a], pts_line_map[b]):
                connected_pairs.append((a, b))
                n_connected += 1
        return n_total, n_connected, connected_pairs

    def _compute_angle(self, line) -> float:
        def angle_for_linestring(ls: LineString) -> Optional[float]:
            coords = list(ls.coords)
            if len(coords) < 2:
                return None
            (x1, y1), (x2, y2) = coords[0], coords[-1]
            dx, dy = x2 - x1, y2 - y1
            return float(np.degrees(np.arctan2(dy, dx)) % 180)

        if isinstance(line, LineString):
            angle = angle_for_linestring(line)
            if angle is None:
                raise ValueError("LineString must have at least two coordinates")
            return angle
        if isinstance(line, MultiLineString):
            angles = [angle_for_linestring(part) for part in line.geoms]
            angles = [a for a in angles if a is not None]
            if not angles:
                raise ValueError("MultiLineString has no valid LineStrings")
            return float(np.mean(angles))
        raise TypeError(f"Unsupported geometry type: {type(line)}")

    def _compute_f1(self, pred: gpd.GeoDataFrame, gt: gpd.GeoDataFrame) -> Tuple[int, int]:
        angle_thres = 30
        match_thres = 10
        num_splits = 5
        tp = 0
        fp = 0
        for _, row in pred.iterrows():
            try:
                geom = row.geometry
                pred_angle = self._compute_angle(geom)
                inter = gt.overlay(
                    gpd.GeoDataFrame(
                        row.to_frame().T.assign(geometry=geom.buffer(self.buffer_size)),
                        geometry="geometry",
                        crs=pred.crs,
                    ),
                    keep_geom_type=True,
                    how="intersection",
                )
                inter["angle"] = inter["geometry"].apply(self._compute_angle)
                inter = inter[inter["angle"].apply(lambda a: abs(a - pred_angle) < angle_thres)]

                split_pts = [geom.interpolate(i / num_splits, normalized=True) for i in range(1, num_splits)]
                if not inter.empty:
                    union_geom = inter.unary_union
                    distances = [pt.distance(union_geom) for pt in split_pts]
                    near = [d for d in distances if d <= match_thres]
                    avg_d = np.average(near) if near else 99999
                    tp += 1 if avg_d < self.e_threshold else 0
                    fp += 0 if avg_d < self.e_threshold else 1
                else:
                    fp += 1
            except Exception:
                fp += 1
        return tp, fp

    def _compute_f1_point_distance(
        self, pred: gpd.GeoDataFrame, gt: gpd.GeoDataFrame, *, dist_thres: float
    ) -> Tuple[int, int]:
        tp = 0
        fp = 0
        for _, row in pred.iterrows():
            try:
                pred_pt = row.geometry
                gt = gt.assign(dist=gt.geometry.distance(pred_pt))
                nearest = float(gt["dist"].min())
                tp += 1 if nearest <= dist_thres else 0
                fp += 0 if nearest <= dist_thres else 1
            except Exception:
                fp += 1
        return tp, fp

    def _get_edge_stats(
        self, polygon: Polygon, G: nx.Graph, gdf: gpd.GeoDataFrame, gdf_gt: gpd.GeoDataFrame
    ) -> Dict[str, object]:
        stats: Dict[str, object] = {}
        undirected_g = nx.Graph(G)
        try:
            n_total, n_conn, pairs = self._tile_traversability_score(undirected_g, polygon)
            stats["n_total_edges"] = n_total
            stats["n_connect_edges"] = n_conn
            stats["connected_pairs"] = " ".join(f"({a},{b})" for a, b in pairs)
        except Exception:
            stats["n_total_edges"] = -99.99
            stats["n_connect_edges"] = -99.99
            stats["connected_pairs"] = "-99.99"
        try:
            tp, fp = self._compute_f1(gdf, gdf_gt)
            _, fn = self._compute_f1(gdf_gt, gdf)
            stats.update({"tp": tp, "fp": fp, "fn": fn})
        except Exception:
            stats.update({"tp": -99.99, "fp": -99.99, "fn": -99.99})
        return stats

    def _get_node_stats(
        self, polygon: Polygon, G: nx.Graph, gdf: gpd.GeoDataFrame, gdf_gt: gpd.GeoDataFrame
    ) -> Dict[str, object]:
        stats: Dict[str, object] = {}
        try:
            tp, fp = self._compute_f1_point_distance(gdf, gdf_gt, dist_thres=self.e_threshold)
            _, fn = self._compute_f1_point_distance(gdf_gt, gdf, dist_thres=self.e_threshold)
            stats.update({"tp": tp, "fp": fp, "fn": fn})
        except Exception:
            stats.update({"tp": -99.99, "fp": -99.99, "fn": -99.99})
        return stats

    def _compute_edge_score(self, feature, gdf, gdf_gt):
        poly = feature.geometry
        if isinstance(poly, (Polygon, MultiPolygon)):
            if isinstance(poly, MultiPolygon) and len(poly.geoms) == 1:
                poly = poly.geoms[0]
            cropped = gpd.clip(gdf, poly)
            cropped_gt = gpd.clip(gdf_gt, poly)
            G = self._graph_from_gdf(cropped)
            m = self._get_edge_stats(poly, G, cropped, cropped_gt)

            # Return exactly the columns declared in meta, plus geometry
            return pd.Series(
                {
                    "geometry": feature.geometry,
                    "total_edges": float(m["n_total_edges"]),
                    "connect_edges": float(m["n_connect_edges"]),
                    "connected_pairs": m["connected_pairs"],
                    "tp": float(m["tp"]),
                    "fp": float(m["fp"]),
                    "fn": float(m["fn"]),
                }
            )

        # If not a polygon, still return the expected keys with nan/empty
        return pd.Series(
            {
                "geometry": feature.geometry,
                "total_edges": np.nan,
                "connect_edges": np.nan,
                "connected_pairs": None,
                "tp": np.nan,
                "fp": np.nan,
                "fn": np.nan,
            }
        )

    def _compute_node_score(self, feature, gdf, gdf_gt):
        poly = feature.geometry
        if isinstance(poly, (Polygon, MultiPolygon)):
            if isinstance(poly, MultiPolygon) and len(poly.geoms) == 1:
                poly = poly.geoms[0]
            cropped = gpd.clip(gdf, poly)
            cropped_gt = gpd.clip(gdf_gt, poly)
            G = self._graph_from_gdf(cropped)
            m = self._get_node_stats(poly, G, cropped, cropped_gt)
            return pd.Series(
                {
                    "geometry": feature.geometry,
                    "tp": float(m["tp"]),
                    "fp": float(m["fp"]),
                    "fn": float(m["fn"]),
                }
            )
        return pd.Series({"geometry": feature.geometry, "tp": np.nan, "fp": np.nan, "fn": np.nan})

    def _join_curb_to_edges(
        self, curb_gdf: gpd.GeoDataFrame, nodes_gdf: gpd.GeoDataFrame, edges_gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Build curb-link nodes by joining curb nodes to edges on both directions.
        Requires columns: _id (nodes), _u_id/_v_id (edges). Keeps last geometry column.
        """
        fwd = pd.merge(curb_gdf, edges_gdf, left_on="_id", right_on="_u_id")
        fwd = pd.merge(fwd, nodes_gdf, left_on="_v_id", right_on="_id", suffixes=("", "_matched"))

        rev = pd.merge(curb_gdf, edges_gdf, left_on="_id", right_on="_v_id")
        rev = pd.merge(rev, nodes_gdf, left_on="_u_id", right_on="_id", suffixes=("", "_matched"))

        curb_link = pd.concat([fwd, rev], ignore_index=True)

        # pick a single geometry column
        geom_cols = [c for c in curb_link.columns if c.startswith("geometry")]
        if len(geom_cols) > 1:
            active = geom_cols[-1]
            curb_link = curb_link.drop([c for c in geom_cols[:-1]], axis=1).rename(columns={active: "geometry"})

        return gpd.GeoDataFrame(curb_link, geometry="geometry", crs=curb_gdf.crs)

    @staticmethod
    def _summary_dict(precision: float, recall: float, f1: float) -> Dict[str, float]:
        return {"precision": precision, "recall": recall, "f1": f1}
