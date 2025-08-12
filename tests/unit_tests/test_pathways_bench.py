import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# ⬇️ if your package path differs, update here
from src.pathways_bench import PathwaysBench
from src.pathways_bench.version import __version__


class TestPathwaysBench(unittest.TestCase):
    # ---- basic init / version ----
    @patch('src.pathways_bench.helpers.MetricsHelper.check_file_exists', autospec=True)
    def test_init_sets_basic_fields_and_version(self, mock_check):
        mock_check.return_value = True
        pb = PathwaysBench(
            gt_file='tests/fixtures/gt.geojson',
            prediction_file='tests/fixtures/pred.geojson',
            output='outdir',
            proj='EPSG:3857',
            debug=True,
        )
        self.assertEqual(pb.PROJ, 'EPSG:3857')
        self.assertTrue(pb.debug)
        self.assertEqual(pb.prediction_file, 'tests/fixtures/pred.geojson')
        self.assertEqual(pb.output, 'outdir')
        self.assertEqual(pb.version, __version__)
        # Do not include the instance (self) when asserting a bound method
        mock_check.assert_called_once_with(filepath='tests/fixtures/gt.geojson')

    # ---- tessellate_area wires Tessellate and stores tip_file ----
    @patch('src.pathways_bench.helpers.MetricsHelper.check_file_exists', autospec=True)
    @patch('src.pathways_bench.Tessellate', autospec=True)
    def test_tessellate_area_calls_tessellate_and_sets_tip_file(self, mock_tessellate_cls, mock_check):
        mock_check.return_value = True

        # Tessellate() instance mock
        tess_inst = mock_tessellate_cls.return_value
        tess_inst.area.return_value = '/tmp/tiles_tip.geojson'

        pb = PathwaysBench(
            gt_file='tests/fixtures/gt.geojson',
            prediction_file='tests/fixtures/pred.geojson',
            output='outdir',
            proj='EPSG:26910',
            debug=False,
        )

        out = pb.tessellate_area(filepath='tests/fixtures/gt.geojson')
        self.assertEqual(out, '/tmp/tiles_tip.geojson')
        self.assertEqual(pb.tip_file, '/tmp/tiles_tip.geojson')

        mock_tessellate_cls.assert_called_once_with(
            filepath='tests/fixtures/gt.geojson',
            proj='EPSG:26910',
            debug=False,
            output='outdir',
            use_pygeos=True,  # default from PathwaysBench
        )
        tess_inst.area.assert_called_once()

        # ---- stats happy path: wires evaluator + reporter and returns reporter output ----
        @patch('src.pathways_bench.helpers.MetricsHelper.check_file_exists', autospec=True)
        @patch('src.pathways_bench.ScoreReporter', autospec=True)
        @patch('src.pathways_bench.GeoStatsEvaluator', autospec=True)
        @patch.object(PathwaysBench, 'tessellate_area', autospec=True)
        def test_stats_happy_path(self, mock_tessellate_area, mock_eval_cls, mock_reporter_cls, mock_check):
            mock_check.return_value = True

            # Make the tessellate mock behave like the real method: set tip_file
            def _fake_tess(self, filepath):
                self.tip_file = '/tmp/tiles_tip.geojson'
                return self.tip_file

            mock_tessellate_area.side_effect = _fake_tess

            # Evaluator instance + run return payload (must include saved_paths)
            eval_inst = mock_eval_cls.return_value
            eval_inst.run.return_value = {
                'saved_paths': {
                    'pred_edge_stats': '/tmp/pred_stats.geojson',
                    'gt_edge_stats': '/tmp/gt_stats.geojson',
                },
                'edge_summary': {'traversability': 0.3, 'precision': 0.5, 'recall': 0.6, 'f1': 0.55},
            }

            # Reporter instance + run return status
            reporter_inst = mock_reporter_cls.return_value
            reporter_inst.run.return_value = {
                'file': 'pred_stats.geojson',
                'mode': 'edge',
                'precision': 0.5,
                'recall': 0.6,
                'f1': 0.55,
                'traversability_similarity': 0.3,
            }

            pb = PathwaysBench(
                gt_file='tests/fixtures/gt.geojson',
                prediction_file='tests/fixtures/pred.geojson',
                output='outdir',
                proj='EPSG:3857',
                debug=True,
            )

            result = pb.stats(threshold=7, buffer_size=9)

            mock_eval_cls.assert_called_once_with(
                proj='EPSG:3857',
                e_threshold=7,
                buffer_size=9,
                output='outdir',
                use_pygeos=True,
            )
            eval_inst.run.assert_called_once_with(
                tile='/tmp/tiles_tip.geojson',
                edges='tests/fixtures/pred.geojson',
                gt_edges='tests/fixtures/gt.geojson',
            )

            mock_reporter_cls.assert_called_once_with(
                pred_path='/tmp/pred_stats.geojson',
                gt_path='/tmp/gt_stats.geojson',
                use_pygeos=True,
            )
            reporter_inst.run.assert_called_once()
            self.assertEqual(result, reporter_inst.run.return_value)

    # ---- stats raises when prediction file missing ----
    @patch('src.pathways_bench.helpers.MetricsHelper.check_file_exists', autospec=True)
    def test_stats_raises_if_prediction_missing(self, mock_check):
        # First call (in __init__, gt) returns True; second call (in stats, pred) raises
        mock_check.side_effect = [True, FileNotFoundError('No such file: pred')]

        pb = PathwaysBench(
            gt_file='tests/fixtures/gt.geojson',
            prediction_file='tests/missing/pred.geojson',
            output=None,
            proj='EPSG:26910',
            debug=False,
        )

        with self.assertRaises(FileNotFoundError):
            _ = pb.stats()

        # gt existence was checked during init, pred during stats
        self.assertEqual(mock_check.call_count, 2)
        # call args: (self, filepath=...)
        self.assertTrue(str(mock_check.call_args_list[0].kwargs['filepath']).endswith('gt.geojson'))
        self.assertTrue(str(mock_check.call_args_list[1].kwargs['filepath']).endswith('pred.geojson'))

    # ---- stats wires tessellate_area before evaluator AND does not hit disk via ScoreReporter ----
    @patch('src.pathways_bench.helpers.MetricsHelper.check_file_exists', autospec=True)
    @patch('src.pathways_bench.ScoreReporter', autospec=True)  # NEW: prevent real file IO
    @patch('src.pathways_bench.GeoStatsEvaluator', autospec=True)
    @patch.object(PathwaysBench, 'tessellate_area', autospec=True)
    def test_stats_calls_tessellate_before_evaluator(self, mock_tessellate_area, mock_eval_cls, mock_reporter_cls, mock_check):
        mock_check.return_value = True

        # Make tessellate set tip_file like real method
        def _fake_tess(self, filepath):
            self.tip_file = '/tmp/tiles_tip.geojson'
            return self.tip_file
        mock_tessellate_area.side_effect = _fake_tess

        # Evaluator returns saved paths (but we won't let ScoreReporter read them)
        eval_inst = mock_eval_cls.return_value
        eval_inst.run.return_value = {
            'saved_paths': {'pred_edge_stats': '/tmp/p.geojson', 'gt_edge_stats': '/tmp/g.geojson'}
        }

        # Reporter is mocked; avoid gpd.read_file in its __init__
        reporter_inst = mock_reporter_cls.return_value
        reporter_inst.run.return_value = {'ok': True}

        pb = PathwaysBench(gt_file='gt.geojson', prediction_file='pred.geojson')
        _ = pb.stats()

        # tessellate_area must be called; it also sets tip_file for real flow
        mock_tessellate_area.assert_called_once_with(pb, filepath='gt.geojson')
        # evaluator.run should have been called once
        eval_inst.run.assert_called_once()
        # reporter constructed with saved paths we provided above
        mock_reporter_cls.assert_called_once_with(
            pred_path='/tmp/p.geojson',
            gt_path='/tmp/g.geojson',
            use_pygeos=True,
        )


if __name__ == '__main__':
    unittest.main()
