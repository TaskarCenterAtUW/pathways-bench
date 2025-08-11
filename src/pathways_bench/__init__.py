from .tessellate import Tessellate
from .version import __version__

from .helpers import check_file_exists
from .geo_evaluator import GeoStatsEvaluator


class PathwaysBench:
    __version__ = __version__

    def __init__(self, gt_file=None, prediction_file=None, output=None, proj='epsg:26910', debug=False):
        self.PROJ = proj
        self.debug = debug
        is_gt_file = check_file_exists(filepath=gt_file)
        if is_gt_file:
            self.gt_file = gt_file

        self.prediction_file = prediction_file
        self.output = output

        self.tip_file = None

    @property
    def version(self):
        return __version__


    def tessellate_area(self, filepath: str):
        tess = Tessellate(filepath=filepath, proj=self.PROJ, debug=self.debug)
        stored_file_path = tess.area()
        self.tip_file = stored_file_path
        return stored_file_path


    def stats(self, threshold=5, buffer_size=5):
        check_file_exists(filepath=self.prediction_file)

        self.tessellate_area(filepath=self.gt_file)

        ev = GeoStatsEvaluator(
            proj=self.PROJ,
            e_threshold=threshold,
            buffer_size=buffer_size,
            output=self.output
        )
        res = ev.run(
            tile=self.tip_file,
            edges=self.prediction_file,
            gt_edges=self.gt_file
        )
        return res