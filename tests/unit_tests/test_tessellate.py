import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import geopandas as gpd
from shapely.geometry import LineString, Polygon, GeometryCollection

from src.pathways_bench.tessellate import Tessellate


def create_dummy_geojson(path):
    data = {
        'type': 'FeatureCollection',
        'features': [
            {
                'type': 'Feature',
                'geometry': {
                    'type': 'LineString',
                    'coordinates': [[-122.42, 37.78], [-122.43, 37.79]]
                },
                'properties': {}
            }
        ]
    }
    with open(path, 'w') as f:
        json.dump(data, f)


class TestTessellate(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.input_path = Path(self.tmpdir.name) / 'input.geojson'
        create_dummy_geojson(self.input_path)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_initialization_loads_file_and_bbox(self):
        tess = Tessellate(filepath=str(self.input_path))
        self.assertIsInstance(tess.gdf, gpd.GeoDataFrame)
        self.assertIsInstance(tess.bbox, Polygon)

    @patch('src.pathways_bench.tessellate.ox.graph.graph_from_polygon')
    def test_osmnx_graph_created(self, mock_graph):
        mock_graph.return_value = 'dummy_graph'
        tess = Tessellate(filepath=str(self.input_path))
        tess._create_osmnx_graph()
        self.assertEqual(tess.g_roads_simplified, 'dummy_graph')

    @patch('src.pathways_bench.tessellate.voronoi_diagram')
    @patch('src.pathways_bench.tessellate.gnx.graph_edges_to_gdf')
    @patch('src.pathways_bench.tessellate.ox.graph.graph_from_polygon')
    def test_voronoi_creation(self, mock_graph, mock_edges, mock_voronoi):
        tess = Tessellate(filepath=str(self.input_path))

        mock_graph.return_value = 'dummy_graph'
        tess.g_roads_simplified = 'dummy_graph'

        dummy_lines = gpd.GeoDataFrame({
            'geometry': [LineString([(-122.42, 37.78), (-122.43, 37.79)])]
        }, crs='EPSG:4326')
        mock_edges.return_value = dummy_lines

        # Create a polygon that overlaps the input geometry
        dummy_poly = Polygon([(-122.44, 37.77), (-122.41, 37.77), (-122.41, 37.80), (-122.44, 37.80)])
        mock_voronoi.return_value = GeometryCollection([dummy_poly])

        tess._create_voronoi_diagram()
        self.assertGreater(len(tess.tile_gdf), 0)

    @patch('src.pathways_bench.tessellate.voronoi_diagram')
    @patch('src.pathways_bench.tessellate.gnx.graph_edges_to_gdf')
    @patch('src.pathways_bench.tessellate.ox.graph.graph_from_polygon')
    def test_area_saves_file_and_returns_path(self, mock_graph, mock_edges, mock_voronoi):
        tess = Tessellate(filepath=str(self.input_path))

        mock_graph.return_value = 'dummy_graph'
        tess.g_roads_simplified = 'dummy_graph'

        dummy_lines = gpd.GeoDataFrame({
            'geometry': [LineString([(-122.42, 37.78), (-122.43, 37.79)])]
        }, crs='EPSG:4326')
        mock_edges.return_value = dummy_lines

        dummy_poly = Polygon([(-122.44, 37.77), (-122.41, 37.77), (-122.41, 37.80), (-122.44, 37.80)])
        mock_voronoi.return_value = GeometryCollection([dummy_poly])

        out_path = tess.area()
        self.assertTrue(Path(out_path).exists())

        gdf_out = gpd.read_file(out_path)
        self.assertFalse(gdf_out.empty)

    @patch('src.pathways_bench.tessellate.os.path.exists', return_value=True)
    @patch('src.pathways_bench.tessellate.gpd.read_file')
    def test_default_output_path_formatting(self, mock_read_file, mock_exists):
        dummy_gdf = gpd.GeoDataFrame({
            'geometry': [LineString([(-1, -1), (1, 1)])]
        }, crs='EPSG:4326')
        mock_read_file.return_value = dummy_gdf

        tess = Tessellate(filepath='/some/path/input.geojson')
        result = tess.filepath.replace('.geojson', '_tip.geojson')
        self.assertEqual(result, '/some/path/input_tip.geojson')

    def test_raises_error_if_file_missing(self):
        with self.assertRaises(FileNotFoundError):
            Tessellate(filepath='/nonexistent/path.geojson')

    def test_raises_error_if_invalid_geojson(self):
        invalid_file = Path(self.tmpdir.name) / 'bad.geojson'
        with open(invalid_file, 'w') as f:
            f.write('invalid content')

        with self.assertRaises(ValueError):
            Tessellate(filepath=str(invalid_file))


    def test_initialization_loads_file_and_bbox_with_debug(self):
        tess = Tessellate(filepath=str(self.input_path), debug=True)
        self.assertIsInstance(tess.gdf, gpd.GeoDataFrame)
        self.assertIsInstance(tess.bbox, Polygon)

    @patch('src.pathways_bench.tessellate.ox.graph.graph_from_polygon')
    def test_osmnx_graph_created_with_debug(self, mock_graph):
        mock_graph.return_value = 'dummy_graph'
        tess = Tessellate(filepath=str(self.input_path), debug=True)
        tess._create_osmnx_graph()
        self.assertEqual(tess.g_roads_simplified, 'dummy_graph')

    @patch('src.pathways_bench.tessellate.voronoi_diagram')
    @patch('src.pathways_bench.tessellate.gnx.graph_edges_to_gdf')
    @patch('src.pathways_bench.tessellate.ox.graph.graph_from_polygon')
    def test_voronoi_creation_with_debug(self, mock_graph, mock_edges, mock_voronoi):
        tess = Tessellate(filepath=str(self.input_path), debug=True)

        mock_graph.return_value = 'dummy_graph'
        tess.g_roads_simplified = 'dummy_graph'

        dummy_lines = gpd.GeoDataFrame({
            'geometry': [LineString([(-122.42, 37.78), (-122.43, 37.79)])]
        }, crs='EPSG:4326')
        mock_edges.return_value = dummy_lines

        # Create a polygon that overlaps the input geometry
        dummy_poly = Polygon([(-122.44, 37.77), (-122.41, 37.77), (-122.41, 37.80), (-122.44, 37.80)])
        mock_voronoi.return_value = GeometryCollection([dummy_poly])

        tess._create_voronoi_diagram()
        self.assertGreater(len(tess.tile_gdf), 0)

    @patch('src.pathways_bench.tessellate.voronoi_diagram')
    @patch('src.pathways_bench.tessellate.gnx.graph_edges_to_gdf')
    @patch('src.pathways_bench.tessellate.ox.graph.graph_from_polygon')
    def test_area_saves_file_and_returns_path_with_debug(self, mock_graph, mock_edges, mock_voronoi):
        tess = Tessellate(filepath=str(self.input_path), debug=True)

        mock_graph.return_value = 'dummy_graph'
        tess.g_roads_simplified = 'dummy_graph'

        dummy_lines = gpd.GeoDataFrame({
            'geometry': [LineString([(-122.42, 37.78), (-122.43, 37.79)])]
        }, crs='EPSG:4326')
        mock_edges.return_value = dummy_lines

        dummy_poly = Polygon([(-122.44, 37.77), (-122.41, 37.77), (-122.41, 37.80), (-122.44, 37.80)])
        mock_voronoi.return_value = GeometryCollection([dummy_poly])

        out_path = tess.area()
        self.assertTrue(Path(out_path).exists())

        gdf_out = gpd.read_file(out_path)
        self.assertFalse(gdf_out.empty)

    @patch('src.pathways_bench.tessellate.os.path.exists', return_value=True)
    @patch('src.pathways_bench.tessellate.gpd.read_file')
    def test_default_output_path_formatting_with_debug(self, mock_read_file, mock_exists):
        dummy_gdf = gpd.GeoDataFrame({
            'geometry': [LineString([(-1, -1), (1, 1)])]
        }, crs='EPSG:4326')
        mock_read_file.return_value = dummy_gdf

        tess = Tessellate(filepath='/some/path/input.geojson', debug=False)
        result = tess.filepath.replace('.geojson', '_tip.geojson')
        self.assertEqual(result, '/some/path/input_tip.geojson')

    def test_raises_error_if_file_missing_with_debug(self):
        with self.assertRaises(FileNotFoundError):
            Tessellate(filepath='/nonexistent/path.geojson', debug=True)

    def test_raises_error_if_invalid_geojson_with_debug(self):
        invalid_file = Path(self.tmpdir.name) / 'bad.geojson'
        with open(invalid_file, 'w') as f:
            f.write('invalid content')

        with self.assertRaises(ValueError):
            Tessellate(filepath=str(invalid_file), debug=True)


if __name__ == '__main__':
    unittest.main()
