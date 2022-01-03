# SRDMD code, data and tutorials
This repository contains code and data used in 
[arXiv:2101.07516](https://arxiv.org/abs/2101.07516) along with tutorials to 
use the code and reproduce some of the results of the paper.

## Note on aliases
We use aliases like `USRLOCAL` throughout for ease. One can define it with
```
export USRLOCAL=/usr/local
```
Running
```
echo $USRLOCAL
```
should then print the path it is aliased to.

If you do not want to use such aliases, you can replace them with the 
corresponding paths in what follows.

## Installation of a Python environment with the prerequisites
We used
[Anaconda 3](https://www.anaconda.com/products/individual),
a Python 3 environment with a scientific stack of libraries.
(Specifically `Anaconda3-2021.11-Linux-x86_64`, which comes with `Python 3.9.7`.)
Assuming this environment is installed to `$ANACONDA3` and is active, 
`which python` should print
```
${ANACONDA3}/bin/python
```
and the directory
```
${ANACONDA3}/include/python3.9
```
should exist. 

We ask users with different Python installations to kindly adapt paths
in the following compilation instructions.

We need the library `pybind11`, which can be installed to an Anaconda environment with
```
conda install pybind11
```

## Installation of the Python wrapper for Channelflow
We used [Channelflow](https://github.com/epfl-ecps/channelflow) for the
direct numerical simulations (DNS) of various channel geometries.
It comes with a Python wrapper, which we changed to use `pybind11`
instead of `Boost.Python`, along with other changes.
As we have not yet had this change merged to upstream yet, there are two
alternatives to work with our changes:

- Apply a patch on a clone of the [official Channelflow](https://github.com/epfl-ecps/channelflow): Copy 
  `channelflow-python-pybind11.diff` from this repository to the directory of the
  clone and inside do
  ```
  git apply channelflow-python-pybind11.diff
  ```
- *Or* use 
  [our fork of Channelflow with this patch already applied](https://github.com/gokhanyalniz/channelflow).

Then please follow [Channelflow's instructions](https://github.com/epfl-ecps/channelflow/blob/master/INSTALL.md) to first install the DNS component.
We had installed its required libraries to `$USRLOCAL`, we list them and
the compiler used here:
- CMake 3.22.1 (`cmake` is on the PATH)
- GCC 11.1 (`g++` is on the PATH)
- FFTW 3.3.10 (file `${USRLOCAL}/lib/libfftw3.a` exists)
- Eigen 3.4.0 (directory `${USRLOCAL}/include/eigen3/Eigen` exists)
- HDF5 1.12.1 (file `${USRLOCAL}/lib/libhdf5.a` exists)
- NetCDF-C 4.8.1 (file `${USRLOCAL}/lib/libnetcdf.a` exists)
- OpenMPI 4.1.2 (`mpic++` is on the PATH)
For compilation commands that we used see [Compiling other software](compiling_other_software).

To compile the Python wrapper component, create a build folder inside the repository, such as `build-python`, and inside the build folder run
```
cmake ../ \
-DCMAKE_BUILD_TYPE=release \
-DCMAKE_CXX_COMPILER=g++ \
-DCMAKE_CXX_FLAGS="-L${USRLOCAL}/lib -lnetcdf -lhdf5_hl -lhdf5 -lz -lcurl" \
-DFFTW_INCLUDE_DIR=${USRLOCAL}/include \
-DFFTW_LIBRARY=${USRLOCAL}/lib/libfftw3.a \
-DWITH_NETCDF=Serial \
-DNETCDF_INCLUDE_DIR=${USRLOCAL}/include \
-DNETCDF_LIBRARY=${USRLOCAL}/lib/libnetcdf.a \
-DEIGEN3_INCLUDE_DIR=${USRLOCAL}/include/eigen3 \
-DUSE_MPI=OFF \
-DWITH_PYTHON=ON \
-DPYTHON_LIBRARY=${ANACONDA3}/lib \
-DPYTHON_EXECUTABLE=${ANACONDA3}/bin/python \
-DPYTHON_INCLUDE_DIR=${ANACONDA3}/include/python3.9 \
-Dpybind11_DIR=${ANACONDA3}/share/cmake/pybind11
make -j libpycf
```
Afterwards, within the build folder, the file `python-wrapper/libpycf.cpython-39-x86_64-linux-gnu.so` should exist.
This is the Python library which we will use to input/output and operate
on Channelflow data within Python scripts.
We will refer to the directory containing this library with `$CFPYTHON`,
that is,
the file `${CFPYTHON}/libpycf.cpython-39-x86_64-linux-gnu.so` exists.
