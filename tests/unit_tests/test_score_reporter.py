import os
import unittest
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
from shapely.geometry import Point
from src.pathways_bench.helpers import MetricsHelper
from src.pathways_bench.score_reporter import ScoreReporter


def gdf_edges_with_pairs(tp=3.0, fp=1.0, fn=2.0, pairs='(0,0) (0,1)'):
    # sums: tp=3, fp=1, fn=2 -> P=0.75, R=0.6, F1≈0.667
    return gpd.GeoDataFrame(
        {'tp': [tp], 'fp': [fp], 'fn': [fn], 'connected_pairs': [pairs], 'geometry': [Point(0, 0)]},
        geometry='geometry',
        crs='EPSG:4326',
    )


def gdf_nodes(tp=2.0, fp=1.0, fn=0.0):
    return gpd.GeoDataFrame(
        {'tp': [tp], 'fp': [fp], 'fn': [fn], 'geometry': [Point(0, 0)]},
        geometry='geometry',
        crs='EPSG:4326',
    )


class TestScoreReporter(unittest.TestCase):
    def setUp(self):
        self._prev_use_pygeos = os.environ.get('USE_PYGEOS')

    def tearDown(self):
        if self._prev_use_pygeos is None:
            os.environ.pop('USE_PYGEOS', None)
        else:
            os.environ['USE_PYGEOS'] = self._prev_use_pygeos

    @patch('src.pathways_bench.score_reporter.gpd.read_file', autospec=True)
    def test_edge_mode_with_ts(self, mock_read):
        pred_path = 'pred_edge_stats.geojson'
        gt_path = 'gt_edge_stats.geojson'

        pred_gdf = gdf_edges_with_pairs(tp=3.0, fp=1.0, fn=2.0, pairs='(0,0) (0,1)')
        gt_gdf = gdf_edges_with_pairs(tp=0.0, fp=0.0, fn=0.0, pairs='(0,0) (1,1)')

        def _rf(path, *args, **kwargs):
            return pred_gdf if str(path).endswith(pred_path) else gt_gdf
        mock_read.side_effect = _rf

        reporter = ScoreReporter(pred_path, gt_path, use_pygeos=False)
        status = reporter.run()

        self.assertEqual(status['file'], Path(pred_path).name)
        self.assertEqual(status['mode'], 'edge')
        self.assertAlmostEqual(status['precision'], 0.75, places=3)
        self.assertAlmostEqual(status['recall'], 0.6, places=3)
        self.assertAlmostEqual(status['f1'], 0.667, places=3)
        self.assertIn('traversability_similarity', status)
        self.assertAlmostEqual(status['traversability_similarity'], 1/3, places=6)
        self.assertEqual(os.environ.get('USE_PYGEOS'), '0')

    @patch('src.pathways_bench.score_reporter.gpd.read_file', autospec=True)
    def test_node_mode_without_pairs(self, mock_read):
        pred_path = 'pred_node_stats.geojson'
        pred_gdf = gdf_nodes(tp=4.0, fp=2.0, fn=2.0)  # P=R=F1=2/3
        mock_read.side_effect = [pred_gdf]

        reporter = ScoreReporter(pred_path, gt_path=None, use_pygeos=True)
        status = reporter.run()

        self.assertEqual(status['mode'], 'node/generic')
        self.assertAlmostEqual(status['precision'], 2/3, places=3)
        self.assertAlmostEqual(status['recall'], 2/3, places=3)
        self.assertAlmostEqual(status['f1'], 2/3, places=3)
        self.assertNotIn('traversability_similarity', status)

    @patch('src.pathways_bench.score_reporter.gpd.read_file', autospec=True)
    def test_edge_mode_without_gt_has_no_ts(self, mock_read):
        pred_path = 'pred_edge_stats.geojson'
        pred_gdf = gdf_edges_with_pairs(tp=1.0, fp=0.0, fn=1.0, pairs='(0,0)')
        mock_read.side_effect = [pred_gdf]

        reporter = ScoreReporter(pred_path, gt_path=None)
        status = reporter.run()

        self.assertEqual(status['mode'], 'edge')
        self.assertNotIn('traversability_similarity', status)

    @patch('src.pathways_bench.score_reporter.gpd.read_file', autospec=True)
    def test_custom_connected_pairs_col_needs_helper_sync_for_ts(self, mock_read):
        pred_path = 'pred_edge_stats.geojson'
        gt_path = 'gt_edge_stats.geojson'

        custom_col = 'pairs_str'
        pred_gdf = gpd.GeoDataFrame(
            {'tp': [3.0], 'fp': [1.0], 'fn': [2.0],
             custom_col: ['(0,0) (0,1)'], 'geometry': [Point(0, 0)]},
            geometry='geometry', crs='EPSG:4326'
        )
        gt_gdf = gpd.GeoDataFrame(
            {'tp': [0.0], 'fp': [0.0], 'fn': [0.0],
             custom_col: ['(0,0) (1,1)'], 'geometry': [Point(0, 0)]},
            geometry='geometry', crs='EPSG:4326'
        )

        def _rf(path, *args, **kwargs):
            return pred_gdf if str(path).endswith(pred_path) else gt_gdf

        mock_read.side_effect = _rf

        reporter = ScoreReporter(pred_path, gt_path, connected_pairs_col=custom_col)

        # First: helper uses non-matching column → no TS
        reporter.metrics.connected_pairs_col = 'connected_pairs'
        status = reporter.run()
        self.assertNotIn('traversability_similarity', status)

        # Then: sync helper to custom column → TS appears
        reporter.metrics.connected_pairs_col = custom_col
        status2 = reporter.run()
        self.assertIn('traversability_similarity', status2)
        self.assertAlmostEqual(status2['traversability_similarity'], 1 / 3, places=6)

    @patch('src.pathways_bench.score_reporter.gpd.read_file', autospec=True)
    def test_init_reads_files(self, mock_read):
        pred_path = 'pred_edge_stats.geojson'
        gt_path = 'gt_edge_stats.geojson'
        mock_read.side_effect = [gdf_edges_with_pairs(), gdf_edges_with_pairs()]

        _ = ScoreReporter(pred_path, gt_path)
        self.assertEqual(mock_read.call_count, 2)
        self.assertTrue(str(mock_read.call_args_list[0][0][0]).endswith(pred_path))
        self.assertTrue(str(mock_read.call_args_list[1][0][0]).endswith(gt_path))

    @patch("src.pathways_bench.score_reporter.gpd.read_file", autospec=True)
    def test_row_ts_uses_matching_index(self, mock_read):
        pred_path = "pred_edge_stats.geojson"
        gt_path = "gt_edge_stats.geojson"

        pred = gpd.GeoDataFrame(
            {"connected_pairs": ["(0,0) (0,1)"]},
            geometry=[Point(0, 0)], crs="EPSG:4326"
        )
        gt = gpd.GeoDataFrame(
            {"connected_pairs": ["(0,0) (1,1)"]},
            geometry=[Point(0, 0)], crs="EPSG:4326"
        )

        def _rf(path, *a, **kw):
            return pred if str(path).endswith(pred_path) else gt

        mock_read.side_effect = _rf

        r = ScoreReporter(pred_path, gt_path)
        # sync helper column to what reporter expects
        r.metrics.connected_pairs_col = r.connected_pairs_col

        # make sure indices align for row lookup
        r.gt_gdf = r.gt_gdf.copy()
        r.gt_gdf.index = r.pred_gdf.index

        row = r.pred_gdf.iloc[0]
        ts = r._row_ts(row, r.gt_gdf)

        # Expected = helper’s aggregate IoU on the same single row
        expected = r.metrics.compute_tra_jaccard(r.pred_gdf.iloc[[0]], r.gt_gdf.iloc[[0]])
        self.assertAlmostEqual(ts, expected, places=6)

    @patch("src.pathways_bench.score_reporter.gpd.read_file", autospec=True)
    def test_row_ts_happy_path_iou_one_third(self, mock_read):
        pred_path = "pred_edge_stats.geojson"
        gt_path = "gt_edge_stats.geojson"

        pred = gpd.GeoDataFrame(
            {"connected_pairs": ["(0,0) (0,1)"], "geometry": [Point(0, 0)]},
            geometry="geometry", crs="EPSG:4326"
        )
        gt = gpd.GeoDataFrame(
            {"connected_pairs": ["(0,0) (1,1)"], "geometry": [Point(0, 0)]},
            geometry="geometry", crs="EPSG:4326"
        )

        def _rf(path, *a, **kw):
            return pred if str(path).endswith(pred_path) else gt

        mock_read.side_effect = _rf

        r = ScoreReporter(pred_path, gt_path)
        # align indices + ensure helper reads same column
        r.gt_gdf = r.gt_gdf.copy()
        r.gt_gdf.index = r.pred_gdf.index
        r.metrics.connected_pairs_col = r.connected_pairs_col

        row = r.pred_gdf.iloc[0]
        ts = r._row_ts(row, r.gt_gdf)
        self.assertAlmostEqual(ts, 1.0 / 3.0, places=6)

    @patch("src.pathways_bench.score_reporter.gpd.read_file", autospec=True)
    def test_row_ts_returns_zero_when_index_missing(self, mock_read):
        pred_path = "pred_edge_stats.geojson"
        gt_path = "gt_edge_stats.geojson"

        pred = gpd.GeoDataFrame(
            {"connected_pairs": ["(0,0) (0,1)"], "geometry": [Point(0, 0)]},
            geometry="geometry", crs="EPSG:4326"
        )
        gt = gpd.GeoDataFrame(
            {"connected_pairs": ["(0,0) (1,1)"], "geometry": [Point(0, 0)]},
            geometry="geometry", crs="EPSG:4326"
        )

        def _rf(path, *a, **kw):
            return pred if str(path).endswith(pred_path) else gt

        mock_read.side_effect = _rf

        r = ScoreReporter(pred_path, gt_path)
        r.metrics.connected_pairs_col = r.connected_pairs_col

        # make gt index not contain pred row index
        bad_gt = r.gt_gdf.copy()
        bad_gt.index = [99]

        row = r.pred_gdf.iloc[0]
        ts = r._row_ts(row, bad_gt)
        self.assertEqual(ts, 0.0)

    @patch("src.pathways_bench.score_reporter.gpd.read_file", autospec=True)
    def test_row_ts_returns_zero_on_missing_column(self, mock_read):
        pred_path = "pred_edge_stats.geojson"
        gt_path = "gt_edge_stats.geojson"

        pred = gpd.GeoDataFrame(
            {"connected_pairs": ["(0,0)"], "geometry": [Point(0, 0)]},
            geometry="geometry", crs="EPSG:4326"
        )
        gt = gpd.GeoDataFrame(
            {"connected_pairs": ["(0,0)"], "geometry": [Point(0, 0)]},
            geometry="geometry", crs="EPSG:4326"
        )

        def _rf(path, *a, **kw):
            return pred if str(path).endswith(pred_path) else gt

        mock_read.side_effect = _rf

        r = ScoreReporter(pred_path, gt_path)
        r.metrics.connected_pairs_col = r.connected_pairs_col
        r.gt_gdf.index = r.pred_gdf.index

        # drop column from the *row* to trigger the except path
        row = r.pred_gdf.iloc[0].copy()
        del row[r.connected_pairs_col]

        ts = r._row_ts(row, r.gt_gdf)
        self.assertEqual(ts, 0.0)

    # ---------- save_scores_geojson ----------

    @patch.object(gpd.GeoDataFrame, "to_file", autospec=True)
    @patch("src.pathways_bench.score_reporter.gpd.read_file", autospec=True)
    def test_save_scores_geojson_writes_default_path_and_adds_ts(self, mock_read, mock_to_file):
        pred_path = "pred_edge_stats.geojson"
        gt_path = "gt_edge_stats.geojson"

        pred = gpd.GeoDataFrame(
            {"connected_pairs": ["(0,0) (0,1)"], "geometry": [Point(0, 0)]},
            geometry="geometry", crs="EPSG:4326"
        )
        gt = gpd.GeoDataFrame(
            {"connected_pairs": ["(0,0) (1,1)"], "geometry": [Point(0, 0)]},
            geometry="geometry", crs="EPSG:4326"
        )

        def _rf(path, *a, **kw):
            return pred if str(path).endswith(pred_path) else gt

        mock_read.side_effect = _rf

        r = ScoreReporter(pred_path, gt_path)
        r.metrics.connected_pairs_col = r.connected_pairs_col
        r.gt_gdf.index = r.pred_gdf.index

        out = r.save_scores_geojson()  # use default name replacement

        # expected filename: *_stats.geojson -> *_scores.geojson
        self.assertTrue(out.endswith("pred_edge_scores.geojson"))
        # TS column present and is 1/3
        self.assertIn("ts", r.pred_gdf.columns)
        self.assertAlmostEqual(float(r.pred_gdf.loc[0, "ts"]), 1.0 / 3.0, places=6)
        # to_file called with the computed path and GeoJSON driver
        mock_to_file.assert_called_once()
        args, kwargs = mock_to_file.call_args
        # first arg is the instance; path is in args[1] due to autospec=True
        self.assertEqual(Path(args[1]).name, "pred_edge_scores.geojson")
        self.assertEqual(kwargs.get("driver"), "GeoJSON")

    @patch.object(gpd.GeoDataFrame, "to_file", autospec=True)
    @patch("src.pathways_bench.score_reporter.gpd.read_file", autospec=True)
    def test_save_scores_geojson_returns_none_when_no_gt(self, mock_read, mock_to_file):
        pred_path = "pred_edge_stats.geojson"
        pred = gpd.GeoDataFrame(
            {"connected_pairs": ["(0,0)"], "geometry": [Point(0, 0)]},
            geometry="geometry", crs="EPSG:4326"
        )
        mock_read.side_effect = [pred]  # only pred, gt=None

        r = ScoreReporter(pred_path, gt_path=None)
        out = r.save_scores_geojson()
        self.assertIsNone(out)
        mock_to_file.assert_not_called()

    @patch.object(gpd.GeoDataFrame, "to_file", autospec=True)
    @patch("src.pathways_bench.score_reporter.gpd.read_file", autospec=True)
    def test_save_scores_geojson_returns_none_when_no_pairs_column(self, mock_read, mock_to_file):
        pred_path = "pred_node_stats.geojson"
        gt_path = "gt_edge_stats.geojson"

        pred_no_pairs = gpd.GeoDataFrame(
            {"tp": [1.0], "fp": [0.0], "fn": [0.0], "geometry": [Point(0, 0)]},
            geometry="geometry", crs="EPSG:4326"
        )
        gt = gpd.GeoDataFrame(
            {"connected_pairs": ["(0,0)"], "geometry": [Point(0, 0)]},
            geometry="geometry", crs="EPSG:4326"
        )

        mock_read.side_effect = [pred_no_pairs, gt]

        r = ScoreReporter(pred_path, gt_path)
        out = r.save_scores_geojson()
        self.assertIsNone(out)
        mock_to_file.assert_not_called()


if __name__ == '__main__':
    unittest.main()
