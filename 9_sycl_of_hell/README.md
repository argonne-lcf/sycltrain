# How to compile

Make
```
CXX=icpx CXXFLAGS=-fsycl make -j

Cmake
```
cmake -S . -B build
cmake --build build/ -j
```

# List of program

- `0_tiny_sycl_info.cpp` How to get information on platform and devices ( `./0_tiny_sycl_info`)
- `1_my_first_kernel.cpp`  How to create queues and command groups ( `./1_my_first_kernel`)
- `2_parallel_for.cpp` How to use `parallel_for` and `range` (`./2_parallel_for -g 8`)
- `3_nd_range`. How to use `nd_range` (`./3_nd_range -g 8 -l 2`)
- `4_memory`  How to allocate memory (`./4_memory -g 8`)
- `5_copy_device_to_host`  How to data-transfer explicitly (`./5_copy_device_to_host -g 8`)
- `6_in_order` How to deal with dependencies: In order (`./6_in_order -g 8 `) 
- `7_out_of_order` How to deal with dependencies: The Correct way (`./7_out_of_order -g 8`)
-  `8_reduction` How to do reduction (`/8_reduction`)
