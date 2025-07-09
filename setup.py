from version import version
from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='pathways_bench',
    version=version,
    author='Sujata Misra',
    author_email='sujatam@gaussiansolutions.com',
    description='A Benchmark for Extracting Routable Pedestrian Path Network Graphs',
    long_description=long_description,
    project_urls={
        'Documentation': 'https://github.com/TaskarCenterAtUW/pathways-bench/blob/main/README.md',
        'GitHub': 'https://github.com/TaskarCenterAtUW/pathways-bench.git',
        'Changelog': 'https://github.com/TaskarCenterAtUW/pathways-bench/blob/main/CHANGELOG.md'
    },
    long_description_content_type='text/markdown',
    url='https://github.com/TaskarCenterAtUW/pathways-bench',
    install_requires=[
        'geopandas==0.14.3',
        'osmnx==1.6.0',
        'geonetworkx==0.5.3',
        'networkx==3.2.1',
        'shapely==2.0.3',
        'fiona==1.9.6'
    ],
    packages=find_packages(where='src'),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
    package_dir={'': 'src'},
)