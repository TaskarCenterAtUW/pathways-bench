import os
from pathways_bench import PathwaysBench

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(PARENT_DIR, f'tests/assets')
GT_DIR = os.path.join(ASSETS_DIR, 'gt')
AI_DIR = os.path.join(ASSETS_DIR, 'predictions_v8')

CITY = 'auburn'
GT_FILE_PATH = f'{GT_DIR}/{CITY}.edges.geojson'
AI_FILE_PATH = f'{AI_DIR}/{CITY}.graph.edges.OSW.geojson'

OUTPUT_DIR = os.path.join(PARENT_DIR, 'output')


def evaluate_test():
    pathways_bench = PathwaysBench(
        gt_file=GT_FILE_PATH,
        prediction_file=AI_FILE_PATH,
        output=OUTPUT_DIR
    )
    results = pathways_bench.stats()
    print('-------')
    print(results)


if __name__ == '__main__':

    evaluate_test()
    # print(f'{os.path.join(AI_OUT_DIR, AI_FILE)}')
    # print(AREA_FILE)

    # TIP -  /Users/anuj/Work/Gaussian/pathways-bench/output/auburn.edges_tip.geojson
    # GT-  /Users/anuj/Work/Gaussian/pathways-bench/tests/assets/gt/auburn.edges.geojson
    # PRED-  /Users/anuj/Work/Gaussian/pathways-bench/tests/assets/predictions/auburn.graph.edges.OSW.geojson
