Usefull link:
https://www.khronos.org/registry/SYCL/specs/sycl-1.2.1.pdf

# 0_tiny_sycl_info.cpp
   
-  ⚛ Print more information. For example max_clock_frequency (or max_compute_units...)
- ⚛⚛⚛ Aply DRY principle to get_info commands. Refractor the code to something like:
```
 for (const auto &type : [name,vendor, version]) {
    plat.get_info<ycl::info::platform::type> 
 }
```

# 1_my_first_kernel.cpp

- ⚛ Run on a diferant platform / device
- ⚛ Add a new kernel
- ⚛⚛ Create a functor for the kernel and launch it twice

# 2_parallel_for.cpp

- ⚛ Add a 2d range
- ⚛ Call function  (example each work item print N `*` where N=<global_index>)

# 4_buffer.cpp
- 1/ Transform the kernel to be a memcopy (add a read buffer)
- 2/ Use another kernel to initilaze the first vector one the device
- 3/ Use directly a vector
- ⚛⚛ Map a 2d data-structure
- ⚛⚛ Copy the data back to the host without destroying the buffer?

# 6_error_handling.cpp
- ⚛⚛⚛ Fix the error

# 8_event.cpp
- ⚛ Use event to do synchornization
