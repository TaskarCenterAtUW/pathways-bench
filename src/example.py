import os
from pathways_bench import PathwaysBench

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(PARENT_DIR, 'tests/assets')
AREA_FILE = os.path.join(ASSETS_DIR, 'portland_footways.geojson')

OUTPUT_DIR = os.path.join(PARENT_DIR, 'output')


def area_test():
    pathways_bench = PathwaysBench()
    print(pathways_bench.version)
    file = pathways_bench.tessellate_area(filepath=AREA_FILE, output_path=f'{OUTPUT_DIR}/output.geojson')
    print(file)


if __name__ == '__main__':
    area_test()