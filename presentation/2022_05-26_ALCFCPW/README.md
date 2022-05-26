# To connect to  theta
```
ssh UUID@theta.alcf.anl.gov
```

# On a theta (non gpu login) node
```
git clone https://github.com/argonne-lcf/sycltrain
```

# To connect to  theta GPU
```
ssh thetagpusn2
```

# On a theta gpu login node (just compile)
```
module use /lus/theta-fs0/software/thetagpu/compilers/llvm/sycl-20220523/modulefiles
module load dpcpp
cd sycltrain/9_sycl_of_hell/
make all
```

# Then On a compute node (just run)
```
qsub -I -q single-gpu -t 60 -n 1 -A Comp_Perf_Workshop
module use /lus/theta-fs0/software/thetagpu/compilers/llvm/sycl-20220523/modulefiles
module load dpcpp
cd ~/sycltrain/9_sycl_of_hell
make run_all
```
