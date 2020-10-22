# ProBLM
C++ implementations of the algorithms introduced in our ACCV 2020 paper:
* Huu Le, Christopher Zach, Edward Rosten, Oliver Woodford, ["Progressive Batching for Efficient Non-linear Least Squares"](https://arxiv.org/pdf/2010.10968.pdf) (Oral)

We are happy to receive any inputs or questions you may have, as well as bug reports. We can be reached by email at huul@chalmers.se. For bug reports, please create issues on this repository.

# Usage 
## Dependencies
The code is written in C++ and depends heavily on the Eigen3 library.
Please make sure that you have Eigen installed. On Ubuntu, this can be done by 
```
sudo apt-get install libeigen3-dev
```

Besides, some functions from the V3D library written by Professor Christopher Zach are also used. 
They are included in this repository (in the Libs folder)

Note that, if you want to compile compile dense homography, libpng and libjpeg are required. These can be installed (on Ubuntu) by:
```
sudo apt-get install libpng-dev libjpeg-dev
```
If you don't want to compile homography, just comment out the last two lines in CMakeLists.txt

## Compile
A CMakelists.txt file has been provided in this repository. From the directory containing the source code, create a build folder. From this build directory, use cmake to generate the makefiles and compile the code:
``` 
mkdir build 
cd build
cmake ..
make -j8
```

## Run the compiled program
### Essential Matrix Fitting
A sample input data is provided in the data folder. This file contains the coordinates of the correspondences extract from a pair of images from the ETH3D dataset (with outliers removed using RANSAC)

To run the program, execute the following command (from the build folder):
```
./problm ../data/putative.txt
```

### Dense Homography 
Two sample images are provided in the data folder. To run dense homography fitting, execute the following command
```
./homography <img1> <img2>
```
For example (from the build folder): 
```
./homography ../data/dino1A.png ../data/dino1B.png
```
