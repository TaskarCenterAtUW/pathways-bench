from .tessellate import Tessellate
from .version import __version__

from .helpers import MetricsHelper
from .geo_evaluator import GeoStatsEvaluator
from .score_reporter import ScoreReporter


class PathwaysBench:
    __version__ = __version__

    def __init__(self, gt_file=None, prediction_file=None, output=None, proj='epsg:26910', debug=False):
        self.PROJ = proj
        self.debug = debug
        self.helper = MetricsHelper()
        is_gt_file = self.helper.check_file_exists(filepath=gt_file)
        if is_gt_file:
            self.gt_file = gt_file

        self.prediction_file = prediction_file
        self.output = output

        self.tip_file = None
        self.use_pygeos = True


    @property
    def version(self):
        return __version__


    def tessellate_area(self, filepath: str):
        tess = Tessellate(filepath=filepath, proj=self.PROJ, debug=self.debug, output=self.output, use_pygeos=self.use_pygeos)
        stored_file_path = tess.area()
        self.tip_file = stored_file_path
        return stored_file_path


    def stats(self, threshold=5, buffer_size=5):
        self.helper.check_file_exists(filepath=self.prediction_file)

        self.tessellate_area(filepath=self.gt_file)

        ev = GeoStatsEvaluator(
            proj=self.PROJ,
            e_threshold=threshold,
            buffer_size=buffer_size,
            output=self.output,
            use_pygeos=self.use_pygeos
        )
        res = ev.run(
            tile=self.tip_file,
            edges=self.prediction_file,
            gt_edges=self.gt_file
        )
        saved_files = res['saved_paths']

        statistic = ScoreReporter(
            pred_path=saved_files['pred_edge_stats'],
            gt_path=saved_files['gt_edge_stats'],
            use_pygeos=self.use_pygeos
        )
        result = statistic.run()

        try:
            scores_path = statistic.save_scores_geojson()
            if scores_path:
                # optionally expose this path in the return payload
                result = dict(result)
                result['scores_path'] = scores_path
        except Exception:
            # never let score-writing break the summary call
            pass
        return result



