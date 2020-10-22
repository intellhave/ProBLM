# ProBLM
C++ implementations of the algorithms introduced in our ACCV 2020 paper:
Huu Le, Christopher Zach, Edward Rosten, Oliver Woodford, ["Progressive Batching for Efficient Non-linear Least Squares"](https://arxiv.org/pdf/2010.10968.pdf) (Oral)


# Usage 
## Dependencies
The code is written in C++ and depends heavily on the Eigen3 library.
Please make sure that you have Eigen installed. On Ubuntu, this can be done by 
```
sudo apt-get install libeigen3-dev
```

## Compile
A CMakelists.txt file has been provided in this repository. From the directory containing the source code, create a build folder. From this build directory, use cmake to generate the makefiles and compile the code:
``` 
mkdir build 
cd build
cmake ..
make -j8
```

## Run the compiled program
A sample input data is provided in the data folder. This file contains the coordinates of the correspondences extract from a pair of images from the ETH3D dataset (with outliers removed using RANSAC)

To run the program, execute the following command (from the build folder):
```
./problm ../data/puttive.txt
```





