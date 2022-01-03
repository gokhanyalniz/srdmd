# Compiling other software
## Setting up aliases
Please adapt the following paths according to your system.

Where to install libraries/compiled binaries to:
```
export USRLOCAL=/home/gokhan/usr/local
```
Various compilers:
```
# C compiler
export CC=/usr/bin/gcc
# C++ compiler
export CXXC=/usr/bin/g++
# Fortran compiler
export FC=/usr/bin/gfortran
```
And their MPI variants:
```
export MCC=/usr/bin/mpicc
export MCXXC=/usr/bin/mpic++
export MFC=/usr/bin/mpifort
```
You will need to update `LD_LIBRARY_PATH` for the resulting compiled binaries
to run:
```
export LD_LIBRARY_PATH=${USRLOCAL}/lib:$LD_LIBRARY_PATH
```
All commands that follow are to be run in the folders extracted from
the downloaded archives, unless otherwise stated.

## [FFTW](https://www.fftw.org/download.html) (with MPI)
```
./configure --prefix=$USRLOCAL --enable-mpi \
CC=$CC \
MPICC=$MCC \
CFLAGS="-fPIC -O3"
make install
```

## [HDF5](https://www.hdfgroup.org/downloads/hdf5) (serial)
```
./configure --prefix=$USRLOCAL \
CC=$CC \
CFLAGS="-fPIC -O3"
make install
```

## [NetCDF-C](https://www.unidata.ucar.edu/downloads/netcdf) (serial)
```
./configure --prefix=$USRLOCAL --disable-dap --disable-shared \
CC=$CC \
CFLAGS="-fPIC -O3 -I${USRLOCAL}/include" \
LDFLAGS=-L${USRLOCAL}/lib
make install
```

## [Eigen](https://gitlab.com/libeigen/eigen/-/releases)
Move the extracted directory to `${USRLOCAL}/include/eigen3`.
It should contain the subdirectory `Eigen`.

## [Channelflow](https://github.com/epfl-ecps/channelflow) (DNS component)
This compiles with FFTW-MPI and serial i/o.

Create a build directory within the extracted archive and go into it.
Point the alias `CHANNELFLOW` to where you would like to install.
Then run
```
cmake ../ \
-DCMAKE_BUILD_TYPE=release \
-DCMAKE_C_COMPILER=$MCC \
-DCMAKE_CXX_COMPILER=$MCXXC \
-DCMAKE_CXX_FLAGS="-L${USRLOCAL}/lib -lnetcdf -lhdf5_hl -lhdf5 -lzip -lcurl" \
-DCMAKE_INSTALL_PREFIX=$CHANNELFLOW \
-DFFTW_INCLUDE_DIR=${USRLOCAL}/include \
-DFFTW_LIBRARY=${USRLOCAL}/lib/libfftw3.a \
-DFFTW_MPI_LIBRARY=${USRLOCAL}/lib/libfftw3_mpi.a \
-DWITH_NETCDF=Serial \
-DNETCDF_INCLUDE_DIR=${USRLOCAL}/include \
-DNETCDF_LIBRARY=${USRLOCAL}/lib/libnetcdf.a \
-DEIGEN3_INCLUDE_DIR=${USRLOCAL}/include/eigen3
make -j install
```
