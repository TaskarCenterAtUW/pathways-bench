# PathwayBench: A Benchmark for Extracting Routable Pedestrian Path Network Graphs

<p align="center"><img width="600" src="./img/teaser.png"></p>


This package contains the PathwayBench dataset and benchmark for extracting routable pedestrian pathway graphs. The dataset includes aerial images, road graphs, road rasters, and ground truth data for multiple cities.

- **Metadata**: [PathwayBench Croissant metadata](./metadata/PathwayBench_metadata.jsonld)
- **License**: [ODbL](https://opendatacommons.org/licenses/odbl/)


## Installation

```shell
# create and activate the conda environment
conda create -n pathwaybench python=3.8
conda activate pathwaybench
Clone the repository and install locally as a package:

```bash
git clone https://github.com/your-username/pathways-bench.git
cd pathways-bench
pip install .
```
This code has been tested with Python 3.10. macOS: Sequoia


## Dependencies

```python
geopandas
shapely
osmnx
geonetworkx
pyproj
```

## Datasets
Each set of samples in the PathwayBench dataset includes five co-registered features. The filename of each set of samples and the corresponding features are listed below:

| Filename | Feature Type
|--|--|
| xxxx_aerial.png | The aerial satellite imagery.
| xxxx_road.geojson | The street (road) graph.
| xxxx_road.png | The rasterized street map (with additional features).
| xxxx_gt_graph.geojson | The human-validated pedestrian pathway graph.
| xxxx_gt_mask.png | The rasterized human-validated pedestrian pathway graph to support semantic segmentation tasks.
| xxxx_gt_color.png | The color-coded version of xxxx_gt_mask.png for visualization purposes.

Below are the links to the dataset that are currently supported by PathwayBench
| City | Data |
|--|--|
| Seattle, WA| [Link to dataset](https://drive.google.com/drive/folders/1CnTVuARwv7j-9WXXJpAb3l6NC3n0nhO9?usp=sharing)
| Washington, D.C. | [Link to dataset](https://drive.google.com/drive/folders/1anMEeDbUZPquwEMGA8V3YPeWQJxFHDnu?usp=sharing)
| Portland, OR | [Link to dataset](https://drive.google.com/drive/folders/1yFViA6PaDxqQWvS_iqDay65pEMijiS05?usp=sharing)
| Bellevue, WA | [Will be released soon]
| Quito, Ecuador | [Will be released soon]
| Sao Paulo, Brazil | [Will be released soon]
| Santiago, Chile | [Will be released soon]  
| Valparaiso, Chile | [Will be released soon]  




## Benchmark

PathwayBench provides utilities for evaluating graphs by the extent to which their structural characteristics align with ground truth, as described below.

Partition test area: This step partitions the entire test area into Tessellating Intersection Polygons (TIP). Each TIP is created by assigning a point location to a road intersection, then computing the associated Voronoi polygons to tessellate the entire test area. `Ground Truth GeoJSON` is provided for each of the support city in PathwayBench dataset.

  ```shell
    from pathways_bench import PathwaysBench
    
    bench = PathwaysBench(proj='epsg:4326', debug=True)
    print(bench.version)
    output_file = bench.tessellate_area(filepath="input.geojson")
    print("Tessellated output saved at:", output_file)
  ```  

### Parameters
- `filepath` - Input GeoJSON defining area of interest (Polygon or MultiPolygon)
- `output_path` - Optional path to save output file
- `proj` - Target projection (default: `epsg:26910`)
- `debug` - Enables detailed logging when set to `True`