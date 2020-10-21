# How to compile

```
make -j 7
```

# List of program

- `tiny_sycl_info.cpp` How to get information on platform and devices ( `./0_tiny_sycl_info`)
- `my_first_kernel.cpp`  How to create queues and command groups ( `./1_my_first_kernel`)
- `parallel_for.cpp` How to use `parallel\_for` and `range` (`./2_parallel_for -g 8`)
- `nd_range`. How to ru se a nd\_range (`./3_nd_range -g 8 -l 2`)
- `buffer`  How to data-transfer (`./4_buffer -g 8`)
- `buffer_update_host`  How to data-transfer explicitly (`./4_buffer -g 8`)
- `error_handling` How to raise Error (`./6_error_handling  1 8 `) # SYCL_PROGRAM_COMPILE_OPTIONS="-cl-std=CL2.0"
- `buffer_usm` How to use one flavor of Unified Shared Memory  (`./7_buffer_usm 8 2`)
- `event` How to use event to get profiling information (`/8_event`)
