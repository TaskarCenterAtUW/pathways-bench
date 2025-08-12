import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
from shapely.geometry import Polygon, LineString, MultiLineString, Point
from src.pathways_bench.geo_evaluator import GeoStatsEvaluator


# ---------- tiny helpers ----------


def tile_one(crs='EPSG:4326'):
    return gpd.GeoDataFrame(
        {'id': [1]},
        geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        crs=crs,
    )

def pred_edges_overlap(crs='EPSG:4326'):
    # one matches GT, one far away
    return gpd.GeoDataFrame(
        {'eid': [1, 2]},
        geometry=[LineString([(0.0, 0.0), (1.0, 0.0)]), LineString([(10.0, 10.0), (11.0, 10.0)])],
        crs=crs,
    )

def gt_edges_overlap(crs='EPSG:4326'):
    # identical to pred[0]
    return gpd.GeoDataFrame(
        {'eid': [99]},
        geometry=[LineString([(0.0, 0.0), (1.0, 0.0)])],
        crs=crs,
    )

def tiles_gdf(crs='EPSG:4326'):
    polys = [
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
    ]
    return gpd.GeoDataFrame({'id': [1, 2]}, geometry=polys, crs=crs)

def edges_gdf_simple(crs='EPSG:4326'):
    lines = [LineString([(0, 0), (0.5, 0.5)]), LineString([(1.5, 0), (1.5, 1)])]
    return gpd.GeoDataFrame({'eid': [10, 11]}, geometry=lines, crs=crs)

def nodes_gdf_simple(crs='EPSG:4326'):
    pts = [Point(0.1, 0.1), Point(1.1, 0.1), Point(1.9, 0.9)]
    return gpd.GeoDataFrame(
        {'_id': [100, 101, 102], 'ext:node_type': ['curb', 'not', 'curb']},
        geometry=pts, crs=crs
    )

def gt_nodes_gdf_simple(crs='EPSG:4326'):
    pts = [Point(0.12, 0.12), Point(1.88, 0.88)]
    return gpd.GeoDataFrame(
        {'_id': [200, 201], 'barrier': ['kerb', 'kerb']},
        geometry=pts, crs=crs
    )

def edges_with_ids(crs='EPSG:4326'):
    # edges link nodes via _u_id/_v_id
    lines = [LineString([(0, 0), (0.1, 0.1)]), LineString([(1.1, 0.1), (1.9, 0.9)])]
    return gpd.GeoDataFrame(
        {'_u_id': [100, 101], '_v_id': [101, 102], 'eid': [500, 501]},
        geometry=lines, crs=crs
    )

def nodes_pred(crs='EPSG:4326'):
    # curb, not curb, curb
    pts = [Point(0.1, 0.1), Point(1.5, 0.5), Point(1.9, 0.9)]
    return gpd.GeoDataFrame(
        {'_id': [100, 101, 102], 'ext:node_type': ['curb', 'not', 'curb']},
        geometry=pts, crs=crs
    )

def nodes_gt(crs='EPSG:4326'):
    # kerb, kerb
    pts = [Point(0.12, 0.12), Point(1.88, 0.88)]
    return gpd.GeoDataFrame(
        {'_id': [200, 201], 'barrier': ['kerb', 'kerb']},
        geometry=pts, crs=crs
    )

def gdf_edges_two(crs="EPSG:4326"):
    e1 = LineString([(0.2,0.2),(1.8,0.2)])
    e2 = LineString([(0.2,1.8),(1.8,1.8)])
    return gpd.GeoDataFrame({"eid":[10,11]}, geometry=[e1,e2], crs=crs)


# ---------- a dask shim so evaluate_* works without dask runtime ----------

class _DummyDaskFrame:
    def __init__(self, gdf):
        self._gdf = gdf

    def apply(self, func, axis, meta, **kwargs):
        # just call pandas apply synchronously
        res = self._gdf.apply(lambda row: func(row, **kwargs), axis=axis)
        class _C:
            def __init__(self, res, crs):
                self._res = res
                self._crs = crs
            def compute(self, scheduler=None):
                # ensure GeoDataFrame with geometry column
                out = gpd.GeoDataFrame(self._res, geometry='geometry', crs=self._crs)
                return out
        return _C(res, self._gdf.crs)


class TestGeoStatsEvaluator(unittest.TestCase):
    def setUp(self):
        self.ev = GeoStatsEvaluator(proj='EPSG:4326', num_partitions=1)

    # -------------------------
    # _coerce_gdf
    # -------------------------
    @patch('src.pathways_bench.geo_evaluator.gpd.read_file', autospec=True)
    def test_coerce_gdf_accepts_path_and_reprojects(self, mock_read):
        src = tiles_gdf(crs='EPSG:4326')
        mock_read.return_value = src
        ev = GeoStatsEvaluator(proj='EPSG:3857')
        out = ev._coerce_gdf('/fake/path.geojson')
        self.assertEqual(str(out.crs).upper(), 'EPSG:3857')
        mock_read.assert_called_once()

    def test_coerce_gdf_accepts_gdf_no_reproject_if_none(self):
        src = tiles_gdf(crs=None)
        ev = GeoStatsEvaluator(proj='EPSG:3857')
        out = ev._coerce_gdf(src)
        self.assertIs(out, src)  # unchanged

    # -------------------------
    # evaluate_edges (dask shim)
    # -------------------------
    @patch('src.pathways_bench.geo_evaluator.dask_geopandas.from_geopandas', autospec=True)
    def test_evaluate_edges_uses_apply_and_returns_gdf(self, mock_from):
        mock_from.side_effect = lambda gdf, npartitions: _DummyDaskFrame(gdf)

        ev = GeoStatsEvaluator(proj='EPSG:4326', num_partitions=1)
        # stub internal scoring to return a minimal Series
        ev._compute_edge_score = MagicMock(side_effect=lambda feature, gdf, gdf_gt: pd.Series({
            'geometry': feature.geometry,
            'total_edges': 10.0,
            'connect_edges': 5.0,
            'connected_pairs': '(0,0) (0,1)',
            'tp': 3.0, 'fp': 1.0, 'fn': 2.0
        }))

        out = ev.evaluate_edges(tiles_gdf(), edges_gdf_simple(), edges_gdf_simple())
        self.assertIsInstance(out, gpd.GeoDataFrame)
        self.assertEqual(len(out), 2)
        self.assertTrue({'total_edges','connect_edges','connected_pairs','tp','fp','fn'} <= set(out.columns))

    # -------------------------
    # run() save-path logic + helper usage
    # -------------------------
    @patch('src.pathways_bench.geo_evaluator.dask_geopandas.from_geopandas', autospec=True)
    @patch.object(gpd.GeoDataFrame, 'to_file', autospec=True)
    @patch('src.pathways_bench.geo_evaluator.gpd.read_file', autospec=True)  # ← add this
    def test_run_generates_output_paths_and_returns_saved_paths(self, mock_read_file, mock_to_file, mock_from):
        mock_from.side_effect = lambda gdf, npartitions: _DummyDaskFrame(gdf)

        # map filenames → tiny GDFs
        def _read_file_side_effect(path, *args, **kwargs):
            path = str(path)
            if path.endswith('tiles.geojson'):
                return tiles_gdf()
            if path.endswith('pred.geojson'):
                return edges_gdf_simple()
            if path.endswith('gt.geojson'):
                return edges_gdf_simple()
            raise FileNotFoundError(path)

        mock_read_file.side_effect = _read_file_side_effect

        ev = GeoStatsEvaluator(proj='EPSG:4326', num_partitions=1, output='outdir/')
        self.assertTrue(ev.output_dir.exists())

        # stub evaluate_edges outputs (so summaries don’t depend on real scoring)
        sample = gpd.GeoDataFrame(
            {'tp': [1.0], 'fp': [0.0], 'fn': [1.0], 'connected_pairs': ['(0,0)']},
            geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])], crs='EPSG:4326'
        )
        ev.evaluate_edges = MagicMock(side_effect=[sample, sample])

        ev.helper.compute_tra_jaccard = MagicMock(return_value=0.42)
        ev.helper.compute_aggregate_f1 = MagicMock(return_value=(0.5, 0.667, 0.571))

        res = ev.run(tile='tiles.geojson', edges='pred.geojson', gt_edges='gt.geojson')

        self.assertIn('saved_paths', res)
        sp = res['saved_paths']
        self.assertTrue(Path(sp['pred_edge_stats']).name.endswith('_stats.geojson'))
        self.assertTrue(Path(sp['gt_edge_stats']).name.endswith('_gt_stats.geojson'))
        self.assertEqual(res['edge_summary']['traversability'], 0.42)
        self.assertEqual(mock_to_file.call_count, 2)

    # -------------------------
    # evaluate_curbs_and_links uses filters and evaluate_nodes
    # -------------------------
    @patch('src.pathways_bench.geo_evaluator.dask_geopandas.from_geopandas', autospec=True)
    def test_evaluate_curbs_and_links_filters_and_calls_evaluate_nodes(self, mock_from):
        mock_from.side_effect = lambda gdf, npartitions: _DummyDaskFrame(gdf)
        ev = GeoStatsEvaluator(proj='EPSG:4326', num_partitions=1)

        # return trivial GDF for evaluate_nodes
        ret = gpd.GeoDataFrame({'tp':[1.0], 'fp':[0.0], 'fn':[0.0]},
                               geometry=[Polygon([(0,0),(1,0),(1,1),(0,1)])], crs='EPSG:4326')
        ev.evaluate_nodes = MagicMock(return_value=ret)

        curbs, curb_links = ev.evaluate_curbs_and_links(
            tiles_gdf(),
            nodes_gdf_simple(),
            gt_nodes_gdf_simple(),
            edges_with_ids(),
            edges_with_ids()
        )
        self.assertIsInstance(curbs, gpd.GeoDataFrame)
        self.assertIsInstance(curb_links, gpd.GeoDataFrame)
        self.assertEqual(list(curbs.columns), ['tp','fp','fn','geometry'])

        # evaluate_nodes should be called twice (curbs and curb_links)
        self.assertEqual(ev.evaluate_nodes.call_count, 2)

    # -------------------------
    # _join_curb_to_edges
    # -------------------------
    def test_join_curb_to_edges_merges_and_keeps_geometry(self):
        ev = GeoStatsEvaluator(proj='EPSG:4326')
        curbs = nodes_gdf_simple()
        nodes_all = curbs.copy()
        edges = edges_with_ids()

        out = ev._join_curb_to_edges(curbs[curbs['ext:node_type']=='curb'], nodes_all, edges)
        self.assertIsInstance(out, gpd.GeoDataFrame)
        self.assertIn('geometry', out.columns)
        self.assertGreaterEqual(len(out), 1)

    # -------------------------
    # summarise_edge_stats / summarise_node_stats (helper passthrough)
    # -------------------------
    def test_summaries_delegate_to_helper(self):
        ev = GeoStatsEvaluator()
        fake_pred = gpd.GeoDataFrame({'tp':[1.0], 'fp':[0.0], 'fn':[1.0], 'connected_pairs':['(0,0)']})
        fake_gt   = gpd.GeoDataFrame({'tp':[1.0], 'fp':[0.0], 'fn':[1.0], 'connected_pairs':['(0,0)']})
        ev.helper.compute_tra_jaccard = MagicMock(return_value=0.9)
        ev.helper.compute_aggregate_f1 = MagicMock(return_value=(0.7, 0.8, 0.75))
        tra, p, r, f1 = ev.summarise_edge_stats(fake_pred, fake_gt)
        self.assertEqual((tra,p,r,f1), (0.9, 0.7, 0.8, 0.75))
        p2, r2, f12 = ev.summarise_node_stats(fake_pred)
        self.assertEqual((p2,r2,f12), (0.7, 0.8, 0.75))

    # -------------------------
    # _compute_angle
    # -------------------------
    def test_compute_angle_linestring_and_multilinestring(self):
        ev = GeoStatsEvaluator()
        ls = LineString([(0,0),(1,1)])
        ang = ev._compute_angle(ls)
        self.assertTrue(0.0 <= ang < 180.0)
        mls = MultiLineString([LineString([(0,0),(1,0)]), LineString([(0,0),(0,1)])])
        ang2 = ev._compute_angle(mls)
        self.assertTrue(0.0 <= ang2 < 180.0)

    # -------------------------
    # evaluate_nodes (Dask shim)
    # -------------------------
    @patch('src.pathways_bench.geo_evaluator.dask_geopandas.from_geopandas', autospec=True)
    def test_evaluate_nodes_returns_gdf(self, mock_from):
        mock_from.side_effect = lambda gdf, npartitions: _DummyDaskFrame(gdf)
        ev = GeoStatsEvaluator(proj='EPSG:4326', num_partitions=1)

        # stub internal per-tile scorer to avoid geometry ops
        ev._compute_node_score = MagicMock(side_effect=lambda feature, gdf, gdf_gt: pd.Series({
            'geometry': feature.geometry, 'tp': 2.0, 'fp': 1.0, 'fn': 0.0
        }))

        out = ev.evaluate_nodes(tiles_gdf(), nodes_pred(), nodes_gt())
        self.assertIsInstance(out, gpd.GeoDataFrame)
        self.assertEqual(len(out), 2)
        self.assertTrue({'tp', 'fp', 'fn'}.issubset(out.columns))

    # -------------------------
    # _compute_f1_point_distance thresholding
    # -------------------------
    def test_compute_f1_point_distance_basic(self):
        ev = GeoStatsEvaluator(proj='EPSG:4326')

        # pred: one near (<=1 deg), one far
        pred = gpd.GeoDataFrame(geometry=[Point(0.0, 0.0), Point(100.0, 100.0)], crs='EPSG:4326')
        gt = gpd.GeoDataFrame(geometry=[Point(0.2, 0.1)], crs='EPSG:4326')

        tp, fp = ev._compute_f1_point_distance(pred, gt, dist_thres=1.0)
        self.assertEqual(tp, 1)
        self.assertEqual(fp, 1)

        # reverse direction should give fn=1 when we treat gt as pred
        tp2, fp2 = ev._compute_f1_point_distance(gt, pred, dist_thres=1.0)
        # both gt points close to first pred point? here only 1 gt point exists → tp2=1, fp2=0
        self.assertEqual((tp2, fp2), (1, 0))

    # -------------------------
    # run() with nodes: saves files & returns summaries
    # -------------------------
    @patch('src.pathways_bench.geo_evaluator.dask_geopandas.from_geopandas', autospec=True)
    @patch.object(gpd.GeoDataFrame, 'to_file', autospec=True)
    @patch('src.pathways_bench.geo_evaluator.gpd.read_file', autospec=True)
    def test_run_with_nodes_saves_paths_and_summaries(self, mock_read, mock_to_file, mock_from):
        mock_from.side_effect = lambda gdf, npartitions: _DummyDaskFrame(gdf)

        # map filenames -> GDFs
        def _read(path, *a, **k):
            path = str(path)
            if path.endswith('tiles.geojson'): return tiles_gdf()
            if path.endswith('pred.geojson'):  return edges_with_ids()
            if path.endswith('gt.geojson'):    return edges_with_ids()
            if path.endswith('nodes.geojson'): return nodes_pred()
            if path.endswith('gtnodes.geojson'): return nodes_gt()
            raise FileNotFoundError(path)

        mock_read.side_effect = _read

        ev = GeoStatsEvaluator(proj='EPSG:4326', num_partitions=1, output='outdir/')

        # edges: return a tiny stats gdf
        edge_stats = gpd.GeoDataFrame(
            {'tp': [1.0], 'fp': [0.0], 'fn': [1.0], 'connected_pairs': ['(0,0)']},
            geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])], crs='EPSG:4326'
        )
        ev.evaluate_edges = MagicMock(side_effect=[edge_stats, edge_stats])

        # nodes: curb_stats and curb_link_stats
        node_stats = gpd.GeoDataFrame(
            {'tp': [1.0], 'fp': [0.0], 'fn': [0.0]},
            geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])], crs='EPSG:4326'
        )
        ev.evaluate_curbs_and_links = MagicMock(return_value=(node_stats, node_stats))

        # summaries
        ev.helper.compute_tra_jaccard = MagicMock(return_value=0.33)
        ev.helper.compute_aggregate_f1 = MagicMock(return_value=(0.5, 0.6, 0.545))

        res = ev.run(
            tile='tiles.geojson',
            edges='pred.geojson',
            gt_edges='gt.geojson',
            nodes='nodes.geojson',
            gt_nodes='gtnodes.geojson',
        )

        # saved paths include node outputs
        self.assertIn('saved_paths', res)
        saved = res['saved_paths']
        self.assertTrue(Path(saved['curb_stats']).name.endswith('_curb_stats.geojson'))
        self.assertTrue(Path(saved['curb_link_stats']).name.endswith('_curb_link_stats.geojson'))
        # and edge files too
        self.assertTrue(Path(saved['pred_edge_stats']).name.endswith('_stats.geojson'))
        self.assertTrue(Path(saved['gt_edge_stats']).name.endswith('_gt_stats.geojson'))

        # summaries present for nodes
        self.assertIn('curb_summary', res)
        self.assertIn('curb_link_summary', res)
        self.assertEqual(res['edge_summary']['traversability'], 0.33)
        # to_file called for 4 outputs (pred, gt, curb, curb_link)
        self.assertEqual(mock_to_file.call_count, 4)

    # -------------------------
    # _compute_node_score wiring
    # -------------------------
    @patch('src.pathways_bench.geo_evaluator.gpd.clip', autospec=True)
    def test__compute_node_score_uses_get_node_stats(self, mock_clip):
        ev = GeoStatsEvaluator(proj='EPSG:4326')
        # tile feature
        tile = tiles_gdf().iloc[0]
        # clip returns the same GDFs (irrelevant for this wiring test)
        clipped_pred = gpd.GeoDataFrame(geometry=[Point(0.1, 0.1)], crs='EPSG:4326')
        clipped_gt = gpd.GeoDataFrame(geometry=[Point(0.12, 0.12)], crs='EPSG:4326')
        mock_clip.side_effect = [clipped_pred, clipped_gt]

        # fake graph and stats
        ev._graph_from_gdf = MagicMock(return_value='G')
        ev._get_node_stats = MagicMock(return_value={'tp': 2, 'fp': 1, 'fn': 0})

        s = ev._compute_node_score(tile, nodes_pred(), nodes_gt())
        self.assertAlmostEqual(s['tp'], 2.0)
        self.assertAlmostEqual(s['fp'], 1.0)
        self.assertAlmostEqual(s['fn'], 0.0)
        ev._graph_from_gdf.assert_called_once()

    # -------------------------
    # save_paths fallback (no output dir, string inputs)
    # -------------------------
    @patch('src.pathways_bench.geo_evaluator.dask_geopandas.from_geopandas', autospec=True)
    @patch.object(gpd.GeoDataFrame, 'to_file', autospec=True)
    @patch('src.pathways_bench.geo_evaluator.gpd.read_file', autospec=True)
    def test_run_save_paths_fallback_with_string_inputs(self, mock_read, mock_to_file, mock_from):
        mock_from.side_effect = lambda gdf, npartitions: _DummyDaskFrame(gdf)

        # simulate disk reads by mapping filenames to tiny GDFs
        def _read(path, *a, **k):
            p = str(path)
            if p.endswith('tiles.geojson'): return tile_one()
            if p.endswith('pred.geojson'):  return pred_edges_overlap()
            if p.endswith('gt.geojson'):    return gt_edges_overlap()
            if p.endswith('nodes.geojson'): return nodes_pred()
            if p.endswith('gtnodes.geojson'): return nodes_gt()
            raise FileNotFoundError(p)

        mock_read.side_effect = _read

        ev = GeoStatsEvaluator(proj='EPSG:4326', num_partitions=1)  # no output dir

        # stub heavy methods so we only test save path logic & summaries
        edge_stats = gpd.GeoDataFrame(
            {'tp': [1.0], 'fp': [0.0], 'fn': [1.0], 'connected_pairs': ['(0,0)']},
            geometry=[tile_one().geometry.iloc[0]], crs='EPSG:4326'
        )
        node_stats = gpd.GeoDataFrame({'tp': [1.0], 'fp': [0.0], 'fn': [0.0]},
                                      geometry=[tile_one().geometry.iloc[0]], crs='EPSG:4326')
        ev.evaluate_edges = MagicMock(side_effect=[edge_stats, edge_stats])
        ev.evaluate_curbs_and_links = MagicMock(return_value=(node_stats, node_stats))
        ev.helper.compute_tra_jaccard = MagicMock(return_value=0.3)
        ev.helper.compute_aggregate_f1 = MagicMock(return_value=(0.5, 0.6, 0.545))

        res = ev.run(
            tile='tiles.geojson',
            edges='pred.geojson',
            gt_edges='gt.geojson',
            nodes='nodes.geojson',
            gt_nodes='gtnodes.geojson',
        )
        self.assertIn('saved_paths', res)
        sp = res['saved_paths']
        # ensure fallback names derived from original filenames
        self.assertTrue(Path(sp['pred_edge_stats']).name == 'pred_stats.geojson')
        self.assertTrue(Path(sp['gt_edge_stats']).name == 'gt_gt_stats.geojson')
        self.assertTrue(Path(sp['curb_stats']).name == 'nodes_curb_stats.geojson')
        self.assertTrue(Path(sp['curb_link_stats']).name == 'nodes_curb_link_stats.geojson')
        # called 4 times (pred, gt, curb, curb_link)
        self.assertEqual(mock_to_file.call_count, 4)

    # -------------------------
    # _add_edges_from_linestring
    # -------------------------
    def test_add_edges_from_linestring_adds_segments_with_attrs(self):
        ev = GeoStatsEvaluator()
        G = nx.Graph()
        ls = LineString([(0,0), (1,0), (1,1)])  # two segments
        ev._add_edges_from_linestring(G, ls, {'eid': 7})
        self.assertEqual(G.number_of_edges(), 2)
        # edges present with attribute
        self.assertEqual(G.edges[(0,0),(1,0)]['eid'], 7)
        self.assertEqual(G.edges[(1,0),(1,1)]['eid'], 7)

    # -------------------------
    # _group_graph_points
    # -------------------------
    def test_group_graph_points_maps_nodes_to_segments(self):
        ev = GeoStatsEvaluator(precision=1e-9)
        G = nx.Graph()
        # nodes exactly on boundary segments of unit square
        pts = [(0.5,0.0), (1.0,0.5), (0.5,1.0), (0.0,0.5)]
        G.add_nodes_from(pts)
        poly = Polygon([(0,0),(1,0),(1,1),(0,1)])
        mapping = ev._group_graph_points(G, poly)
        # bottom=0,right=1,top=2,left=3 given polygon coordinate order
        self.assertIn((0.5,0.0), mapping[0])
        self.assertIn((1.0,0.5), mapping[1])
        self.assertIn((0.5,1.0), mapping[2])
        self.assertIn((0.0,0.5), mapping[3])

    # -------------------------
    # _edges_are_connected
    # -------------------------
    def test_edges_are_connected_true_and_false(self):
        ev = GeoStatsEvaluator()
        G = nx.Graph()
        a = (0.5,0.0); b = (1.0,0.5); c = (0.0,0.5)
        G.add_nodes_from([a,b,c])
        G.add_edge(a,b)
        # a↔b path exists
        self.assertTrue(ev._edges_are_connected(G, [a], [b]))
        # a↛c no path
        self.assertFalse(ev._edges_are_connected(G, [a], [c]))

    # -------------------------
    # _tile_traversability_score
    # -------------------------
    def test_tile_traversability_score_counts_connected_pairs(self):
        ev = GeoStatsEvaluator(precision=1e-9)
        poly = Polygon([(0,0),(1,0),(1,1),(0,1)])
        # build a graph with two boundary nodes connected
        G = nx.Graph()
        a = (0.5,0.0)  # bottom edge
        b = (1.0,0.5)  # right edge
        G.add_edge(a, b)
        # monkeypatch grouping to only two segments so totals are predictable
        with patch.object(ev, '_group_graph_points', return_value={0: [a], 1: [b]}):
            n_total, n_conn, pairs = ev._tile_traversability_score(G, poly)
        # combinations_with_replacement of {0,1} -> (0,0), (0,1), (1,1) -> 3
        self.assertEqual(n_total, 3)
        self.assertEqual(n_conn, 1)
        self.assertEqual(pairs, [(0,1)])

    # -------------------------
    # _compute_f1 simple functional test
    # -------------------------
    def test_compute_f1_tp_and_fp(self):
        ev = GeoStatsEvaluator(proj='EPSG:4326', buffer_size=0.01, e_threshold=5.0)
        tp, fp = ev._compute_f1(pred_edges_overlap(), gt_edges_overlap())
        # first pred line matches → tp=1; second is far → fp=1
        self.assertEqual((tp, fp), (1, 1))

    # -------------------------
    # _get_edge_stats wiring
    # -------------------------
    def test_get_edge_stats_wires_traversability_and_f1(self):
        ev = GeoStatsEvaluator()
        poly = tile_one().geometry.iloc[0]
        G = nx.Graph()

        with patch.object(ev, '_tile_traversability_score', return_value=(10, 2, [(0,0)])), \
             patch.object(ev, '_compute_f1', side_effect=[(3,1), (999, 4)]):  # second call supplies fn
            stats = ev._get_edge_stats(poly, G, pred_edges_overlap(), gt_edges_overlap())

        self.assertEqual(stats['n_total_edges'], 10)
        self.assertEqual(stats['n_connect_edges'], 2)
        self.assertEqual(stats['connected_pairs'], '(0,0)')
        self.assertEqual(stats['tp'], 3)
        self.assertEqual(stats['fp'], 1)
        self.assertEqual(stats['fn'], 4)


    # -------------------------
    # _get_node_stats wiring
    # -------------------------
    def test_get_node_stats_uses_point_distance(self):
        ev = GeoStatsEvaluator()
        poly = tile_one().geometry.iloc[0]
        G = nx.Graph()
        with patch.object(ev, '_compute_f1_point_distance', side_effect=[(5,2), (999, 3)]):
            stats = ev._get_node_stats(poly, G, nodes_pred(), nodes_gt())
        self.assertEqual(stats['tp'], 5)
        self.assertEqual(stats['fp'], 2)
        self.assertEqual(stats['fn'], 3)

    def test_graph_from_gdf_with_linestring_and_multilinestring(self):
        # One LineString with 3 points -> 2 segments
        ls = LineString([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])
        # One MultiLineString with two 2-point LineStrings -> 2 segments
        mls = MultiLineString([
            LineString([(2.0, 0.0), (2.0, 1.0)]),
            LineString([(2.0, 1.0), (3.0, 1.0)]),
        ])
        gdf = gpd.GeoDataFrame(
            {'eid': [1, 2], 'geometry': [ls, mls]},
            geometry='geometry',
            crs='EPSG:4326'
        )

        ev = GeoStatsEvaluator(proj='EPSG:4326')
        G = ev._graph_from_gdf(gdf)

        # Type & counts
        self.assertIsInstance(G, nx.Graph)
        # 2 segments from ls + 2 segments from mls = 4 edges
        self.assertEqual(G.number_of_edges(), 4)

        # Attributes copied to each edge created from a row
        self.assertEqual(G.edges[(0.0, 0.0), (1.0, 0.0)]['eid'], 1)
        self.assertEqual(G.edges[(1.0, 0.0), (1.0, 1.0)]['eid'], 1)
        self.assertEqual(G.edges[(2.0, 0.0), (2.0, 1.0)]['eid'], 2)
        self.assertEqual(G.edges[(2.0, 1.0), (3.0, 1.0)]['eid'], 2)

        # Nodes are coordinate tuples
        self.assertIn((0.0, 0.0), G.nodes)
        self.assertIn((3.0, 1.0), G.nodes)

    def test_graph_from_gdf_empty(self):
        empty = gpd.GeoDataFrame({'eid': []}, geometry=gpd.GeoSeries([], dtype='geometry'), crs='EPSG:4326')
        ev = GeoStatsEvaluator(proj='EPSG:4326')
        G = ev._graph_from_gdf(empty)
        self.assertEqual(G.number_of_nodes(), 0)
        self.assertEqual(G.number_of_edges(), 0)

    @patch('src.pathways_bench.geo_evaluator.gpd.clip', autospec=True)
    def test_compute_edge_score_happy_path(self, mock_clip):
        # Arrange tile feature (as the dask/pandas row would look)
        tile_poly = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        feature = pd.Series({'geometry': tile_poly})

        # Return cropped copies (content is irrelevant because we mock downstream)
        cropped = gdf_edges_two()
        cropped_gt = gdf_edges_two()
        mock_clip.side_effect = [cropped, cropped_gt]

        # mock internal helpers
        fake_G = MagicMock()
        self.ev._graph_from_gdf = MagicMock(return_value=fake_G)
        self.ev._get_edge_stats = MagicMock(return_value={
            'n_total_edges': 7, 'n_connect_edges': 3,
            'connected_pairs': '(0,0) (0,1)', 'tp': 5, 'fp': 2, 'fn': 1
        })

        # Act
        out = self.ev._compute_edge_score(feature, gdf_edges_two(), gdf_edges_two())

        # Assert: returns a Series with exact keys/values and carries geometry through
        self.assertIsInstance(out, pd.Series)
        self.assertEqual(out['geometry'], tile_poly)
        self.assertEqual(float(out['total_edges']), 7.0)
        self.assertEqual(float(out['connect_edges']), 3.0)
        self.assertEqual(out['connected_pairs'], '(0,0) (0,1)')
        self.assertEqual(float(out['tp']), 5.0)
        self.assertEqual(float(out['fp']), 2.0)
        self.assertEqual(float(out['fn']), 1.0)

        self.ev._graph_from_gdf.assert_called_once()
        self.ev._get_edge_stats.assert_called_once()

    def test_compute_edge_score_non_polygon_returns_nans(self):
        # Arrange: feature geometry is a Point, so branch should return NaNs/None
        feature = pd.Series({'geometry': Point(0, 0)})

        out = self.ev._compute_edge_score(feature, gdf_edges_two(), gdf_edges_two())

        self.assertTrue(pd.isna(out['total_edges']))
        self.assertTrue(pd.isna(out['connect_edges']))
        self.assertIsNone(out['connected_pairs'])
        self.assertTrue(pd.isna(out['tp']))
        self.assertTrue(pd.isna(out['fp']))
        self.assertTrue(pd.isna(out['fn']))

if __name__ == '__main__':
    unittest.main()
