# MParT: A Monotone Parameterization Toolbox
A CPU/GPU performance-portable library for parameterizing and constructing monotone functions in the context of measure transport and regression.

## Installation
Instructions will go here...

### Compiling
MParT uses CMake to handle dependencies and compiler configurations.   A basic build of MParT that should work on most operating systems can be obtained with:
```
mkdir build
cd build
cmake                           \
  -DKokkos_ENABLE_PTHREAD=ON    \
  -DKokkos_ENABLE_SERIAL=ON     \
..
make
```
This will compile the `mpart` library and also create a test executable called `RunTests`.  The tests can be run with:
```
./RunTests
```
Or, with the additional specification of the number of Kokkos threads to use:
```
./RunTests --kokkos-threads=4
```

#### Options:
MParT is built on Kokkos, which provides a single interface to many different multithreading capabilities like pthreads, OpenMP, CUDA, and OpenCL.   A list of available backends can be found on the [Kokkos wiki](https://github.com/kokkos/kokkos/blob/master/BUILD.md#device-backends).   The `Kokkos_ENABLE_PTHREAD` option in the CMake configuration above can be changed to reflect different choices in device backends.   The OSX-provided clang compiler does not support OpenMP, so `PTHREAD` is a natural choice for CPU-based multithreading on OSX.   However, you may find that OpenMP has slightly better performance with other compilers and operating systems.


## Example Usage
Provide a short introductory example to hook users...

## Citing
How should users cite this package?

## Contributing
How do we want people contribute to MPart?   Fork and pull request?
