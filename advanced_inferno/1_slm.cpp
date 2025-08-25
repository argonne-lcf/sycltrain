#include <stdio.h>

#include <iostream>
#include <sycl/sycl.hpp>
#include "argparse.hpp"

int get_values( int * A_ptr, int global_range, int local_range, int x, int y ) {
  if( y == 0 || y == (local_range-1)) {
    return 0;
  }
  return A_ptr[std::clamp(0, (x - 1), global_range-1)] + A_ptr[x] +
    A_ptr[std::clamp(0, (x + 1), global_range-1)];
}

int main(int argc, char **argv) {
  //  _                ___
  // |_) _. ._ _  _     |  ._  ._     _|_
  // |  (_| | _> (/_   _|_ | | |_) |_| |_
  //
  argparse::ArgumentParser program("1_slm");

  program.add_argument("-g", "--global")
      .help("Global Range")
      .default_value(1)
      .action([](const std::string &value) { return std::stoi(value); });

  program.add_argument("-l", "--local")
      .help("Local Range")
      .default_value(1)
      .action([](const std::string &value) { return std::stoi(value); });

  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error &err) {
    std::cerr << err.what() << std::endl;
    std::cout << program;
    std::exit(1);
  }

  const auto global_range = program.get<int>("-g");
  const auto local_range = program.get<int>("-l");
  
  sycl::queue Q;
  
  size_t local_mem_size = Q.get_device().get_info<sycl::info::device::local_mem_size>();
  std::cout << "Local Memory Size Per Workgroup: " << local_mem_size << " bytes" << std::endl;

  //    __                                                           
  //   (_  |_   _. ._ _    |   _   _  _. |   |\/|  _  ._ _   _  ._   
  //   __) | | (_| | (/_   |_ (_) (_ (_| |   |  | (/_ | | | (_) | \/ 
  //                                                              /  

  // Create custom shared usm allocator and use it
  sycl::usm_allocator<int, sycl::usm::alloc::shared> allocator(Q);
  std::vector<int, decltype(allocator)> A(global_range, allocator);
  std::iota(A.begin(), A.end(), 0);
  std::vector<int> A_ref(A.begin(), A.end());
  
  auto A_ptr = A.data();
  Q.submit([&](sycl::handler &cgh) {
    // note that here we allocate all available SLM. if we increase this by 1 it will fail at
    // runtime with an UR_RESULT_ERROR_OUT_OF_RESOURCES error
    sycl::local_accessor<int> tmp_wg(sycl::range<1>(local_mem_size / sizeof(int)), cgh);

    cgh.parallel_for(
         sycl::nd_range<1>(global_range, local_range), [=](sycl::nd_item<1> i) {
           int x = i.get_global_linear_id();
           int y = i.get_local_linear_id();
           tmp_wg[y] = get_values( A_ptr, global_range, local_range, x,y);
           i.barrier();
	   A_ptr[x] = tmp_wg[y];
         });
   }).wait();

  for (int i=0;i<global_range;i++) {
    int ref = get_values( A_ref.data(), global_range, local_range, i, i %local_range);
    assert( A[i] == ref);
     
  }
  printf("Success!\n");
  
  return 0;
}
