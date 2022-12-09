# Symmetry-reduced Dynamic Mode Decomposition of Near-wall Turbulence: Code and data
This repository contains code and data used in 
[arXiv:2101.07516](https://arxiv.org/abs/2101.07516), to appear in the 
Journal of Fluid Mechanics, along with tutorials to use the code and reproduce 
some of the results of the paper.

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

We need the libraries `pybind11` and `pyvista`.
We installed `pybind11` with `conda`
```
conda install pybind11
```
and installed `pyvista` with `pip`
```
pip install pyvista
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
the compilers used here:
- CMake 3.22.1 (`cmake` is on the PATH)
- GCC 11.1 (`g++` is on the PATH)
- FFTW 3.3.10 (file `${USRLOCAL}/lib/libfftw3.a` exists)
- Eigen 3.4.0 (directory `${USRLOCAL}/include/eigen3/Eigen` exists)
- HDF5 1.12.1 (file `${USRLOCAL}/lib/libhdf5.a` exists)
- NetCDF-C 4.8.1 (file `${USRLOCAL}/lib/libnetcdf.a` exists)
- OpenMPI 4.1.2 (`mpic++` is on the PATH)

For the compilation commands that we used for these see [Compiling other software](compiling_other_software.md).

To compile the Python wrapper component, create a build folder inside the repository, such as `build-python` and go into it.
Point the alias `CHANNELFLOW_PYTHON` to where you would like to install, we 
will refer to this alias in the Python scripts as well.
Then run
```
cmake ../ \
-DCMAKE_BUILD_TYPE=release \
-DCMAKE_CXX_COMPILER=g++ \
-DCMAKE_CXX_FLAGS="-L${USRLOCAL}/lib -lnetcdf -lhdf5_hl -lhdf5 -lzip -lcurl" \
-DCMAKE_INSTALL_PREFIX=$CHANNELFLOW_PYTHON \
-DFFTW_INCLUDE_DIR=${USRLOCAL}/include \
-DFFTW_LIBRARY=${USRLOCAL}/lib/libfftw3.a \
-DWITH_NETCDF=Serial \
-DNETCDF_INCLUDE_DIR=${USRLOCAL}/include \
-DNETCDF_LIBRARY=${USRLOCAL}/lib/libnetcdf.a \
-DEIGEN3_INCLUDE_DIR=${USRLOCAL}/include/eigen3 \
-DUSE_MPI=OFF \
-DWITH_PYTHON=ON \
-DWITH_GTEST=OFF \
-DPYTHON_LIBRARY=${ANACONDA3}/lib \
-DPYTHON_EXECUTABLE=${ANACONDA3}/bin/python \
-DPYTHON_INCLUDE_DIR=${ANACONDA3}/include/python3.9 \
-Dpybind11_DIR=${ANACONDA3}/share/cmake/pybind11
make -j install
```
Afterwards, the file
```
${CHANNELFLOW_PYTHON}/lib/libpycf.cpython-39-x86_64-linux-gnu.so
```
should exist.
Note that this library is linked to the files 
`libchflow.so` and `libnsolver.so` in the same directory.
You will need to point `LD_LIBRARY_PATH` to this directory if 
you move it and keep `libpycf-*.so` separate.

### libstdc++ errors
When you do `import libpycf` in Python, if you encounter an error of the sort
```
ImportError: ../anaconda3/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.29' not found (required by ../channelflow-python/lib/libpycf.cpython-39-x86_64-linux-gnu.so)
```
find where `libstdc++.so.6` is in your system _outside_ the Python environment.
In Ubuntu this should be at
```
/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```
in Arch Linux at
```
/usr/lib/libstdc++.so.6
```
We will use this to replace the copy in the Python environment.
First backup the Python environment's copy:
```
mv -vf ${ANACONDA3}/lib/libstdc++.so.6 ${ANACONDA3}/lib/libstdc++.so.6.old
```
Then, create a link to the system copy at the Python environment.
Using the Ubuntu path as an example, that can be done with
```
ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ${ANACONDA3}/lib/libstdc++.so.6
```