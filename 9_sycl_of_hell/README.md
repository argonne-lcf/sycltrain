# How to compile

```
make -j 7
```

# List of program

- `tiny_sycl_info.cpp` How to get information on platform and devices ( `./0_tiny_sycl_info`)
- `my_first_kernel.cpp`  How to create queues and command groups ( `./1_my_first_kernel`)
- `parallel_for.cpp` How to use a parralle\_for and range (`./2_parallel_for 8`)
- `nd_range`. How to ru se a nd\_range (`./3_nd_range 8 2`)
- `buffer`  How to data-transfer (`./4_buffer 8 1`)
- `matrix_mul_local_intel` Local Shared Memory by Intel.
- `error_handling` How to raise Error (`./6_error_handling  1 8 `)
- `buffer_usm` How to use one flavor of Unified shared memory  (`./7_buffer_usm 8 2`)
- `simple dag` Demonstration of a DAG. Warning lot of C++ (`./8_simple_dag 10`)
