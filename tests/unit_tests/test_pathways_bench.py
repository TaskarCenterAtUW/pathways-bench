import unittest
from unittest.mock import patch, MagicMock

from src.pathways_bench import PathwaysBench
from src.pathways_bench.version import __version__


class TestPathwaysBench(unittest.TestCase):

    def test_version_property(self):
        bench = PathwaysBench()
        self.assertEqual(bench.version, __version__)

    def test_initialization_defaults(self):
        bench = PathwaysBench()
        self.assertEqual(bench.PROJ, 'epsg:26910')
        self.assertFalse(bench.debug)

    def test_initialization_custom_values(self):
        bench = PathwaysBench(proj='epsg:3857')
        self.assertEqual(bench.PROJ, 'epsg:3857')

    def test_initialization_with_debug(self):
        bench = PathwaysBench(proj='epsg:3857', debug=True)
        self.assertEqual(bench.PROJ, 'epsg:3857')
        self.assertTrue(bench.debug)

    @patch('src.pathways_bench.Tessellate')
    def test_tessellate_area_calls_tessellate_with_correct_args(self, mock_tessellate):
        mock_instance = MagicMock()
        mock_instance.area.return_value = '/fake/output.geojson'
        mock_tessellate.return_value = mock_instance

        bench = PathwaysBench(proj='epsg:4326')
        result = bench.tessellate_area(filepath='/fake/input.geojson', output_path='/fake/output.geojson')

        mock_tessellate.assert_called_once_with(
            filepath='/fake/input.geojson',
            proj='epsg:4326',
            debug=False
        )
        mock_instance.area.assert_called_once_with(out_path='/fake/output.geojson')
        self.assertEqual(result, '/fake/output.geojson')

    @patch('src.pathways_bench.Tessellate')
    def test_tessellate_area_with_default_output_path(self, mock_tessellate):
        mock_instance = MagicMock()
        mock_instance.area.return_value = '/fake/input_tip.geojson'
        mock_tessellate.return_value = mock_instance

        bench = PathwaysBench()
        result = bench.tessellate_area(filepath='/fake/input.geojson')

        mock_tessellate.assert_called_once_with(
            filepath='/fake/input.geojson',
            proj='epsg:26910',
            debug=False
        )
        mock_instance.area.assert_called_once_with(out_path=None)
        self.assertEqual(result, '/fake/input_tip.geojson')

    @patch('src.pathways_bench.Tessellate')
    def test_tessellate_area_propagates_exception(self, mock_tessellate):
        mock_tessellate.side_effect = ValueError('Invalid input file')

        bench = PathwaysBench()
        with self.assertRaises(ValueError) as cm:
            bench.tessellate_area(filepath='/bad/path.geojson')

        self.assertEqual(str(cm.exception), 'Invalid input file')

    @patch('src.pathways_bench.Tessellate')
    def test_tessellate_area_calls_tessellate_with_correct_args_with_debug(self, mock_tessellate):
        mock_instance = MagicMock()
        mock_instance.area.return_value = '/fake/output.geojson'
        mock_tessellate.return_value = mock_instance

        bench = PathwaysBench(proj='epsg:4326', debug=True)
        result = bench.tessellate_area(filepath='/fake/input.geojson', output_path='/fake/output.geojson')

        mock_tessellate.assert_called_once_with(
            filepath='/fake/input.geojson',
            proj='epsg:4326',
            debug=True
        )
        mock_instance.area.assert_called_once_with(out_path='/fake/output.geojson')
        self.assertEqual(result, '/fake/output.geojson')

    @patch('src.pathways_bench.Tessellate')
    def test_tessellate_area_with_default_output_path_with_debug(self, mock_tessellate):
        mock_instance = MagicMock()
        mock_instance.area.return_value = '/fake/input_tip.geojson'
        mock_tessellate.return_value = mock_instance

        bench = PathwaysBench()
        result = bench.tessellate_area(filepath='/fake/input.geojson')

        mock_tessellate.assert_called_once_with(
            filepath='/fake/input.geojson',
            proj='epsg:26910',
            debug=False
        )
        mock_instance.area.assert_called_once_with(out_path=None)
        self.assertEqual(result, '/fake/input_tip.geojson')

    @patch('src.pathways_bench.Tessellate')
    def test_tessellate_area_propagates_exception_with_debug(self, mock_tessellate):
        mock_tessellate.side_effect = ValueError('Invalid input file')

        bench = PathwaysBench()
        with self.assertRaises(ValueError) as cm:
            bench.tessellate_area(filepath='/bad/path.geojson')

        self.assertEqual(str(cm.exception), 'Invalid input file')


if __name__ == '__main__':
    unittest.main()
