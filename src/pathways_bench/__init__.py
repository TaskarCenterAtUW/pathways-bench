from .tessellate import Tessellate
from .version import __version__

class PathwaysBench:
    __version__ = __version__

    def __init__(self, proj='epsg:26910', debug=False):
        self.PROJ = proj
        self.debug = debug

    @property
    def version(self):
        return __version__


    def tessellate_area(self, filepath: str, output_path=None):
        tess = Tessellate(filepath=filepath, proj=self.PROJ, debug=self.debug)
        stored_file_path = tess.area(out_path=output_path)
        return stored_file_path


