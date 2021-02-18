[![Build Status](https://travis-ci.com/basselin7u/GPU-Restricted-Power-Diagrams.svg?token=6EKdueDyx6sBdzqppzDs&branch=main)](https://travis-ci.com/basselin7u/GPU-Restricted-Power-Diagrams)
[![GitHub](https://img.shields.io/github/license/basselin7u/GPU-Restricted-Power-Diagrams)](https://opensource.org/licenses/MIT)

# Restricted Power Diagrams on the GPU

The following code is the implementation of the Restricted Power Diagrams on the GPU article published at Eurographics in 2021 (link to come).
This method simultaneously decomposes a 3D object into power diagram cells and integrates given functions in each of the obtained simple regions, on the GPU.

To build under Linux or macOS:

```sh
mkdir build
cd build/
cmake ..
make
cd ..
```

Three sets of 100K points are provided (blue noise, perturbed grid, white noise).
Following commands compute Voronoi diagrams restricted to a volume (elephant):

```sh
build/test_voronoi --grid=data/gpu_elephant_100.txt --output=xyz --K=20 --P=25 --V=30 data/100K-blue.xyz
build/test_voronoi --grid=data/gpu_elephant_100.txt --output=xyz --K=50 --P=30 --V=40 data/100K-perturbed-grid.xyz
build/test_voronoi --grid=data/gpu_elephant_100.txt --output=xyz --K=70 --P=30 --V=40 data/100K-white.xyz
```

Each command produces the file gridGPU.xyz; the file contains barycenters of non-empty cells.

NOTE:
these sources contain the float and the double implementations of restricted voronoi,
- to compile the float's implementation, use cmake -DUSE_DOUBLE:BOOL=OFF ..
- to compile the double's implementation, use cmake -DUSE_DOUBLE:BOOL=ON ..


The misc folder contains some useful code for this project:
- createGrid: code to create the input grid file of cVoroOpenCL
- rvd: code to launch geogram's RVD on a volume ( given a list of seed)
- diffResult: code to do a diff of two ascii result files with format
  numPoints
  m0*x0 m0*y0 m0*z0 m0
  m1*x1 m1*y1 m1*z1 m1
  m2*x2 m2*y2 m2*z2 m2
  ...

To compile this directory, you must first add the geogram's source in it. Then, you
can do ./configure.sh, ...

Examples:
```sh
createGrid division=50 input.geogram grid=input_50.grid output=finalInput.geogram
rvd input=finalInput.geogram points=20Knormed.xyz output=rvd_mg.xyzw
diffResult rvd_mg.xyzw gridGPUres.xyzw
```


