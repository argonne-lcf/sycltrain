# To connect
```
ssh UUID@theta.alcf.anl.gov
ssh thetagpusn2
```

# On a theta (non gpu login) node
```
git clone https://github.com/argonne-lcf/sycltrain
```

# On a theta gpu login node (just compile)
```
module use /soft/thetagpu/compilers/dpcpp/modulefiles
module use /lus/theta-fs0/software/environment/thetagpu/lmod/modulefiles
module load dpcpp
module load nvhpc/21.2
clang++ -std=c++17 -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -Wno-unknown-cuda-version FILENAME.cpp
#or in `~/sycltrain/9_sycl_of_hell`

CXX=clang++ CXXFLAGS="-std=c++17 -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -fsycl-unnamed-lambda -Wno-unknown-cuda-version" make all -k
```

# On a compute node (just run)
```
qsub -t 60 -n 1 -q single-gpu -A ATPESC2021 -I
module use /soft/modulefiles/
module load dpcpp
```
