import os
os.environ['USE_PYGEOS'] = '0'

from pathlib import Path
from shapely.geometry import Polygon
from shapely.ops import voronoi_diagram
import geopandas as gpd
import osmnx as ox
import geonetworkx as gnx
from .logger import get_logger


class Tessellate:
    def __init__(self, filepath: str, proj='epsg:26910', debug=False):
        self.filepath = filepath
        self.PROJ = proj
        self.logger = get_logger(self.__class__.__name__, debug)

        if not os.path.exists(filepath):
            self.logger.error(f"File not found: {filepath}")
            raise FileNotFoundError(f"No such file: {filepath}")

        try:
            self.logger.debug('Reading input file...')
            self.gdf = gpd.read_file(filepath)
        except Exception as e:
            self.logger.error(f"Failed to read GeoJSON: {e}")
            raise ValueError("Invalid GeoJSON input") from e

        self.logger.debug('Generating bounding box from input data...')
        self.bbox = self._bounding_box_from_gdf(self.gdf)
        self.g_roads_simplified = None
        self.tile_gdf = None

    def _bounding_box_from_gdf(self, gdf: gpd.GeoDataFrame) -> Polygon:
        min_x, min_y, max_x, max_y = gdf.total_bounds
        return Polygon([
            (min_x, min_y),
            (max_x, min_y),
            (max_x, max_y),
            (min_x, max_y)
        ])

    def _create_osmnx_graph(self):
        self.logger.debug('Downloading and simplifying OSM graph...')
        self.g_roads_simplified = ox.graph.graph_from_polygon(
            self.bbox, network_type='drive', simplify=True, retain_all=True
        )

    def _create_voronoi_diagram(self):
        self.logger.debug('Creating Voronoi diagram...')
        gdf_roads = gnx.graph_edges_to_gdf(self.g_roads_simplified)
        voronoi = voronoi_diagram(gdf_roads.boundary.unary_union, envelope=self.bbox)
        voronoi_gdf = gpd.GeoDataFrame({'geometry': voronoi.geoms}, crs=gdf_roads.crs)
        clipped = gpd.clip(voronoi_gdf, self.bbox)
        self.tile_gdf = clipped.to_crs(self.PROJ)
        self.logger.debug('Voronoi diagram created and projected.')

    def area(self, out_path: str = None):
        self.logger.info('Starting tessellation process...')
        self._create_osmnx_graph()
        self._create_voronoi_diagram()

        if out_path is None:
            out_path = self.filepath.replace('.geojson', '_tip.geojson')

        self.logger.info(f'Saving output to {out_path}')
        self.tile_gdf.to_file(out_path, driver='GeoJSON')
        return out_path

