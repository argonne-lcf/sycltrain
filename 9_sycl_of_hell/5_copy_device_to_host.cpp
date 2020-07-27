#include "cxxopts.hpp"
#include <CL/sycl.hpp>
#include <vector>
#include <algorithm>    // std::copy

namespace sycl = cl::sycl;

int main(int argc, char **argv) {

  //  _                ___
  // |_) _. ._ _  _     |  ._  ._     _|_
  // |  (_| | _> (/_   _|_ | | |_) |_| |_
  //

  cxxopts::Options options("4_buffer", " How to use 'nd_range' ");

  options.add_options()("h,help", "Print help")(
      "g,grange", "Global Range", cxxopts::value<int>()->default_value("1"));

  auto result = options.parse(argc, argv);

  if (result.count("help")) {
    std::cout << options.help({"", "Group"}) << std::endl;
    exit(0);
  }

  const auto global_range = result["grange"].as<int>();
  //  _       _   _
  // |_)    _|_ _|_ _  ._
  // |_) |_| |   | (/_ |
  //

  // Create array
  std::vector<int> A(global_range);
  // The buffer is created outside of the scope
  // Need to manualy update it
  sycl::buffer<sycl::cl_int, 1> bufferA(A.data(), A.size());
  
  // Selectors determine which device kernels will be dispatched to.
  sycl::default_selector selector;
  // Create your own or use `{cpu,gpu,accelerator}_selector`
  {
    sycl::queue myQueue(selector);
    std::cout << "Running on "
              << myQueue.get_device().get_info<sycl::info::device::name>()
              << "\n";

    // Create a command_group to issue command to the group
    myQueue.submit([&](sycl::handler &cgh) {
      // Create an accesor for the sycl buffer. Trust me, use auto.
      auto accessorA = bufferA.get_access<sycl::access::mode::discard_write>(cgh);
      // Nd range allow use to access information
      cgh.parallel_for<class hello_world>(
          sycl::range<1>{sycl::range<1>(global_range)},
          [=](sycl::id<1> idx) {
           accessorA[idx]=idx[0];
          }); // End of the kernel function
    });       // End of the queue commands
  
    // Now update the host buffer 
    myQueue.submit([&](sycl::handler &cgh) {
      auto accessorA = bufferA.get_access<sycl::access::mode::read>(cgh);
      cgh.copy(accessorA,A.data());
      // If you can prove that `bufferA` have no internal copy, this should work too
      //cgh.update_host(accessorA);
    });
    // The synchronization append at the buffer destructor, not the queue.
    // Hence we need to explicitly wait
    myQueue.wait();

  } // End of scope.
  for (size_t i = 0; i < global_range; i++)
    std::cout << "A[ " << i << " ] = " << A[i] << std::endl;
  return 0;
}
