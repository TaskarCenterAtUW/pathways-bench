import os

from pathlib import Path
from shapely.geometry import Polygon, MultiLineString
from shapely.ops import voronoi_diagram
import geopandas as gpd
import osmnx as ox
import geonetworkx as gnx
from .logger import get_logger


class Tessellate:
    def __init__(
            self,
            filepath: str,
            proj='epsg:26910',
            debug=False,
            output: str = None,
            use_pygeos: bool = False,
    ):
        if not use_pygeos:
            os.environ['USE_PYGEOS'] = '0'

        # normalize to Path
        self.filepath = Path(filepath)
        self.PROJ = proj
        self.logger = get_logger(self.__class__.__name__, debug)

        self.output_dir = Path(output) if output else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # existence check via Path
        if not self.filepath.exists():
            self.logger.error(f'File not found: {self.filepath}')
            raise FileNotFoundError(f'No such file: {self.filepath}')

        try:
            self.logger.debug('Reading input file...')
            # pass str(path) to geopandas
            self.gdf = gpd.read_file(str(self.filepath))
        except Exception as e:
            self.logger.error(f'Failed to read GeoJSON: {e}')
            raise ValueError('Invalid GeoJSON input') from e

        self.logger.debug('Generating bounding box from input data...')
        self.boundary = self._outer_boundary_from_gdf(self.gdf)
        self.g_roads_simplified = None
        self.tile_gdf = None

    def _outer_boundary_from_gdf(self, gdf: gpd.GeoDataFrame) -> Polygon:
        """
        Make a convex-hull boundary from the input geometries.
        Matches original `tessellate_area.create_tip` logic.
        """
        try:
            # Try to mirror the original: MultiLineString of all geometries → convex_hull
            multi_line = MultiLineString(list(gdf.geometry.values))
            return multi_line.convex_hull
        except Exception:
            # Fallback: robust union → convex_hull (safer for mixed geom types)
            return gdf.unary_union.convex_hull

    def _create_osmnx_graph(self):
        self.logger.debug('Downloading and simplifying OSM graph...')
        self.g_roads_simplified = ox.graph.graph_from_polygon(
            self.boundary, network_type='drive', simplify=True, retain_all=True
        )

    def _create_voronoi_diagram(self):
        self.logger.debug('Creating Voronoi diagram...')
        gdf_roads = gnx.graph_edges_to_gdf(self.g_roads_simplified)
        voronoi = voronoi_diagram(gdf_roads.boundary.unary_union, envelope=self.boundary)
        voronoi_gdf = gpd.GeoDataFrame({'geometry': list(voronoi.geoms)}, crs=gdf_roads.crs)
        clipped = gpd.clip(voronoi_gdf, self.boundary)
        self.tile_gdf = clipped.to_crs(self.PROJ)
        self.logger.debug('Voronoi diagram created and projected.')

    def area(self):
        self.logger.info('Starting tessellation process...')
        self._create_osmnx_graph()
        self._create_voronoi_diagram()

        if self.output_dir is None:
            out_path = self.filepath.with_name(self.filepath.stem + '_tip.geojson')
        else:
            filename = self.filepath.stem + '_tip.geojson'
            out_path = self.output_dir / filename

        self.logger.info(f'Saving output to {out_path}')
        self.tile_gdf.to_file(str(out_path), driver='GeoJSON')
        return str(out_path)
