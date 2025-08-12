import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import geopandas as gpd
from shapely.geometry import Polygon, LineString, MultiLineString

# ⬇️ change this import to your actual module path
from src.pathways_bench.tessellate import Tessellate


def _dummy_input_gdf():
    # simple square polygon
    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    return gpd.GeoDataFrame({'id': [1]}, geometry=[poly], crs='EPSG:4326')

def gdf_lines_square(crs='EPSG:4326'):
    # 2 edges of a right triangle (0,0)-(2,0) and (2,0)-(1,1)
    ls1 = LineString([(0, 0), (2, 0)])
    ls2 = LineString([(2, 0), (1, 1)])
    return gpd.GeoDataFrame({'id': [1, 2]}, geometry=[ls1, ls2], crs=crs)


class TestTessellate(unittest.TestCase):
    def setUp(self):
        # make a temp file path that 'exists' for the constructor check
        self.tmpdir = tempfile.TemporaryDirectory()
        self.input_path = Path(self.tmpdir.name) / 'input.geojson'
        self.input_path.write_text('{}')  # contents don't matter; we'll patch read_file

    def tearDown(self):
        self.tmpdir.cleanup()

    # -----------------------
    # __init__ error handling
    # -----------------------

    def test_init_raises_when_file_missing(self):
        missing = Path(self.tmpdir.name) / 'nope.geojson'
        with self.assertRaises(FileNotFoundError):
            Tessellate(filepath=str(missing))

    def test_init_raises_on_invalid_geojson(self):
        # Patch gpd.read_file to raise an exception (simulate parse error)
        with patch('src.pathways_bench.tessellate.gpd.read_file', side_effect=Exception('bad input')):
            with self.assertRaises(ValueError):
                Tessellate(filepath=str(self.input_path))

    # -----------------------
    # _bounding_box_from_gdf
    # -----------------------

    def test_bounding_box_from_gdf(self):
        with patch('src.pathways_bench.tessellate.gpd.read_file', return_value=_dummy_input_gdf()):
            t = Tessellate(filepath=str(self.input_path))
            # bbox should match our dummy polygon bounds
            minx, miny, maxx, maxy = t.gdf.total_bounds
            bb = t._outer_boundary_from_gdf(t.gdf)
            self.assertTrue(bb.equals(Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])))

    # -----------------------
    # area() path logic & saving
    # -----------------------

    def test_area_writes_default_path_when_no_output_dir(self):
        # 1) Patch read_file to supply a small input GDF
        with patch('src.pathways_bench.tessellate.gpd.read_file', return_value=_dummy_input_gdf()):
            t = Tessellate(filepath=str(self.input_path))

        # 2) Stub heavy internals and supply a minimal tile_gdf (projected is fine to reuse same)
        t._create_osmnx_graph = MagicMock()
        t._create_voronoi_diagram = MagicMock()

        # 3) Provide a minimal tile_gdf and intercept to_file to avoid disk IO
        t.tile_gdf = _dummy_input_gdf()
        with patch.object(gpd.GeoDataFrame, 'to_file', autospec=True) as mock_to_file:
            out_path = t.area()
            # default naming: replace .geojson with _tip.geojson
            expected = str(t.filepath).replace('.geojson', '_tip.geojson')
            self.assertEqual(str(out_path), expected)
            mock_to_file.assert_called_once()  # saved

    def test_area_writes_into_output_dir_when_given(self):
        outdir = Path(self.tmpdir.name) / 'tiles'
        # 1) Patch read_file to supply input
        with patch('src.pathways_bench.tessellate.gpd.read_file', return_value=_dummy_input_gdf()):
            t = Tessellate(filepath=str(self.input_path), output=str(outdir))

        # ensure directory created
        self.assertTrue(outdir.exists())

        # 2) Stub heavy internals
        t._create_osmnx_graph = MagicMock()
        t._create_voronoi_diagram = MagicMock()

        # 3) Provide tile_gdf and intercept to_file
        t.tile_gdf = _dummy_input_gdf()
        with patch.object(gpd.GeoDataFrame, 'to_file', autospec=True) as mock_to_file:
            out_path = t.area()
            # output filename should be <original_file_name>_tip.geojson in output dir
            expected_name = t.filepath.name.replace('.geojson', '_tip.geojson')
            self.assertEqual(Path(out_path).name, expected_name)
            self.assertEqual(Path(out_path).parent, outdir)
            mock_to_file.assert_called_once()

    def test_area_calls_internal_steps(self):
        with patch('src.pathways_bench.tessellate.gpd.read_file', return_value=_dummy_input_gdf()):
            t = Tessellate(filepath=str(self.input_path))

        with patch.object(t, '_create_osmnx_graph') as mock_graph, \
             patch.object(t, '_create_voronoi_diagram') as mock_voronoi, \
             patch.object(gpd.GeoDataFrame, 'to_file', autospec=True):
            # need tile_gdf ready before saving; simulate that _create_voronoi_diagram sets it
            def _fake_voronoi():
                t.tile_gdf = _dummy_input_gdf().to_crs(t.PROJ)
            mock_voronoi.side_effect = _fake_voronoi

            _ = t.area()

            mock_graph.assert_called_once()
            mock_voronoi.assert_called_once()

    # -----------------------
    # output dir creation
    # -----------------------

    def test_output_dir_created_on_init(self):
        outdir = Path(self.tmpdir.name) / 'new_output_dir'
        with patch('src.pathways_bench.tessellate.gpd.read_file', return_value=_dummy_input_gdf()):
            _ = Tessellate(filepath=str(self.input_path), output=str(outdir))
        self.assertTrue(outdir.exists())

    # ---------------------------
    # _create_osmnx_graph
    # ---------------------------

    @patch('src.pathways_bench.tessellate.ox.graph.graph_from_polygon')
    @patch('src.pathways_bench.tessellate.gpd.read_file', autospec=True)
    def test_create_osmnx_graph_sets_graph(self, mock_read_file, mock_graph_from_polygon):
        # Arrange
        mock_read_file.return_value = _dummy_input_gdf()
        mock_graph_from_polygon.return_value = 'DUMMY_GRAPH'

        t = Tessellate(filepath=str(self.input_path))  # PROJ default ok
        # Act
        t._create_osmnx_graph()

        # Assert
        self.assertEqual(t.g_roads_simplified, 'DUMMY_GRAPH')
        mock_graph_from_polygon.assert_called_once()
        # verify bbox passed and kwargs used
        args, kwargs = mock_graph_from_polygon.call_args
        self.assertIs(args[0], t.boundary)
        self.assertEqual(kwargs['network_type'], 'drive')
        self.assertTrue(kwargs['simplify'])
        self.assertTrue(kwargs['retain_all'])

    # ---------------------------
    # _create_voronoi_diagram
    # ---------------------------

    @patch('src.pathways_bench.tessellate.gpd.clip', autospec=True)
    @patch('src.pathways_bench.tessellate.voronoi_diagram', autospec=True)
    @patch('src.pathways_bench.tessellate.gnx.graph_edges_to_gdf', autospec=True)
    @patch('src.pathways_bench.tessellate.gpd.read_file', autospec=True)
    def test_create_voronoi_diagram_sets_tile_gdf(
            self, mock_read_file, mock_graph_edges_to_gdf, mock_voronoi, mock_clip
    ):
        # Arrange: init with PROJ same as source CRS to avoid real reprojection
        mock_read_file.return_value = _dummy_input_gdf()  # EPSG:4326
        t = Tessellate(filepath=str(self.input_path), proj='EPSG:4326')

        # Provide a stub graph first so edges_to_gdf can consume it
        t.g_roads_simplified = MagicMock(name='dummy_graph')

        # Return a tiny edges GeoDataFrame with CRS
        edges_gdf = gpd.GeoDataFrame(
            {'eid': [1, 2]},
            geometry=[LineString([(0, 0), (1, 1)]), LineString([(1, 0), (2, 1)])],
            crs='EPSG:4326',
        )
        mock_graph_edges_to_gdf.return_value = edges_gdf

        # Dummy voronoi return: needs .geoms -> list of polygons
        class DummyVoronoi:
            def __init__(self, geoms):
                self.geoms = geoms

        dummy_polys = [Polygon([(0, 0), (2, 0), (1, 1)])]
        mock_voronoi.return_value = DummyVoronoi(dummy_polys)

        # Clip returns a GeoDataFrame built from those polygons with same CRS
        clipped_gdf = gpd.GeoDataFrame({'geometry': dummy_polys}, geometry='geometry', crs=edges_gdf.crs)
        mock_clip.return_value = clipped_gdf

        # Act
        t._create_voronoi_diagram()

        # Assert: tile_gdf created and projected to t.PROJ
        self.assertIsInstance(t.tile_gdf, gpd.GeoDataFrame)
        self.assertEqual(len(t.tile_gdf), 1)
        self.assertEqual(str(t.tile_gdf.crs).upper(), 'EPSG:4326')

        # Verify the call chain
        mock_graph_edges_to_gdf.assert_called_once_with(t.g_roads_simplified)
        mock_voronoi.assert_called_once()
        mock_clip.assert_called_once()

        # Verify voronoi called with unary_union boundary and bbox
        v_args, v_kwargs = mock_voronoi.call_args
        # first positional arg is a geometry (boundary.unary_union), not easily comparable
        self.assertIn('envelope', v_kwargs)
        self.assertIs(v_kwargs['envelope'], t.boundary)

    @patch('src.pathways_bench.tessellate.get_logger', autospec=True)
    @patch.object(Path, 'exists', autospec=True)
    @patch('src.pathways_bench.tessellate.gpd.read_file', autospec=True)
    def test_outer_boundary_from_gdf_convex_hull(self, mock_read, mock_exists, _mock_logger):
        # Arrange: fake file exists + return simple lines
        mock_exists.return_value = True
        src = gdf_lines_square()
        mock_read.return_value = src

        # Instantiate Tessellate with dummy path (no actual I/O)
        t = Tessellate(filepath='/tmp/dummy.geojson', proj='EPSG:4326', debug=False, output=None, use_pygeos=True)

        # Expected outer boundary = convex hull of all lines in gdf
        multi = MultiLineString(list(src.geometry.values))
        expected_hull = multi.convex_hull  # a Polygon

        # Act
        # assuming the private exists; if it's static/instance the call is the same
        hull = t._outer_boundary_from_gdf(src)

        # Assert
        self.assertTrue(hull.equals(expected_hull))
        self.assertEqual(hull.geom_type, 'Polygon')


if __name__ == '__main__':
    unittest.main()
