import os
import tempfile
import unittest
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from src.pathways_bench.helpers import MetricsHelper, CONNECTED_PAIRS_COL


class TestMetricsHelper(unittest.TestCase):
    def setUp(self):
        self.helper = MetricsHelper(connected_pairs_col=CONNECTED_PAIRS_COL)

    # ---------- check_file_exists ----------

    def test_check_file_exists_ok(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b'ok')
            tmp_path = tmp.name
        try:
            self.assertTrue(MetricsHelper.check_file_exists(tmp_path))
        finally:
            os.unlink(tmp_path)

    def test_check_file_exists_missing(self):
        missing = os.path.join(tempfile.gettempdir(), 'this_file_should_not_exist_12345.txt')
        if os.path.exists(missing):
            os.remove(missing)
        with self.assertRaises(FileNotFoundError):
            MetricsHelper.check_file_exists(missing)

    # ---------- compute_aggregate_f1 ----------

    def test_compute_aggregate_f1_basic(self):
        # tp=3, fp=1, fn=1 -> precision=0.75, recall=0.75, f1=0.75
        gdf = gpd.GeoDataFrame({'tp': [1, 1, 1], 'fp': [0, 1, 0], 'fn': [0, 0, 1]})
        p, r, f1 = self.helper.compute_aggregate_f1(gdf)
        self.assertAlmostEqual(p, 0.75, places=3)
        self.assertAlmostEqual(r, 0.75, places=3)
        self.assertAlmostEqual(f1, 0.75, places=3)

    def test_compute_aggregate_f1_handles_missing_columns(self):
        # No tp/fp/fn columns present -> treated as zeros
        gdf = gpd.GeoDataFrame({'something_else': [1, 2, 3]})
        p, r, f1 = self.helper.compute_aggregate_f1(gdf)
        self.assertEqual((p, r, f1), (0.0, 0.0, 0.0))

    def test_compute_aggregate_f1_with_nans(self):
        # sums: tp=3, fp=1, fn=1 -> p=0.75 r=0.75 f1=0.75
        gdf = gpd.GeoDataFrame({'tp': [1, np.nan, 2], 'fp': [0, 1, np.nan], 'fn': [0, 0, 1]})
        p, r, f1 = self.helper.compute_aggregate_f1(gdf)
        self.assertAlmostEqual(p, 0.75, places=3)
        self.assertAlmostEqual(r, 0.75, places=3)
        self.assertAlmostEqual(f1, 0.75, places=3)

    def test_compute_aggregate_f1_divide_by_zero(self):
        gdf = gpd.GeoDataFrame({'tp': [0], 'fp': [0], 'fn': [0]})
        p, r, f1 = self.helper.compute_aggregate_f1(gdf)
        self.assertEqual((p, r, f1), (0.0, 0.0, 0.0))

    # ---------- compute_tra_jaccard ----------

    def test_compute_tra_jaccard_simple(self):
        # Row1: identical -> IoU 1.0
        # Row2: partial overlap {(0,1)} vs {(0,1),(2,2)} -> IoU 1/2 = 0.5
        pred = gpd.GeoDataFrame({CONNECTED_PAIRS_COL: ['(0,0) (1,1)', '(0,1)']})
        gt   = gpd.GeoDataFrame({CONNECTED_PAIRS_COL: ['(0,0) (1,1)', '(0,1) (2,2)']})
        val = self.helper.compute_tra_jaccard(pred, gt)
        self.assertAlmostEqual(val, (1.0 + 0.5) / 2.0, places=9)

    def test_compute_tra_jaccard_filters_sentinel(self):
        # First row sentinel -> ignored; second row identical -> IoU 1.0
        pred = gpd.GeoDataFrame({CONNECTED_PAIRS_COL: ['-99.99', '(3,4)']})
        gt   = gpd.GeoDataFrame({CONNECTED_PAIRS_COL: ['-99.99', '(3,4)']})
        val = self.helper.compute_tra_jaccard(pred, gt)
        self.assertAlmostEqual(val, 1.0, places=9)

    def test_compute_tra_jaccard_all_ignored_returns_zero(self):
        pred = gpd.GeoDataFrame({CONNECTED_PAIRS_COL: ['-99.99', '-99.99']})
        gt   = gpd.GeoDataFrame({CONNECTED_PAIRS_COL: ['-99.99', '-99.99']})
        val = self.helper.compute_tra_jaccard(pred, gt)
        self.assertEqual(val, 0.0)

    def test_compute_tra_jaccard_empty_sets(self):
        # Empty/blank strings -> treated as empty sets; skipped; average => 0.0
        pred = gpd.GeoDataFrame({CONNECTED_PAIRS_COL: ['', '  ']})
        gt   = gpd.GeoDataFrame({CONNECTED_PAIRS_COL: ['', '']})
        val = self.helper.compute_tra_jaccard(pred, gt)
        self.assertEqual(val, 0.0)

    def test_compute_tra_jaccard_malformed_tokens(self):
        # Malformed strings should not crash; treated as empty -> result 0.0
        pred = gpd.GeoDataFrame({CONNECTED_PAIRS_COL: ['foo bar', '(1,)']})
        gt   = gpd.GeoDataFrame({CONNECTED_PAIRS_COL: ['baz', '(1,2,3)']})
        val = self.helper.compute_tra_jaccard(pred, gt)
        self.assertEqual(val, 0.0)

    def test_connected_pairs_col_override(self):
        # Prove we can use a different column name via constructor
        helper2 = MetricsHelper(connected_pairs_col='pairs')
        pred = gpd.GeoDataFrame({'pairs': ['(1,2) (3,4)']})
        gt   = gpd.GeoDataFrame({'pairs': ['(1,2)']})
        val = helper2.compute_tra_jaccard(pred, gt)
        # IoU: {(1,2)} vs {(1,2),(3,4)} => 1/2
        self.assertAlmostEqual(val, 0.5, places=9)



if __name__ == '__main__':
    unittest.main()
