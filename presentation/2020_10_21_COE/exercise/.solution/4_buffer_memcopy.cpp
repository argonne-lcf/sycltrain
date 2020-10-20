#include "argparse.hpp"
#include <CL/sycl.hpp>
#include <vector>

namespace sycl = cl::sycl;

int main(int argc, char **argv) {

  //  _                ___
  // |_) _. ._ _  _     |  ._  ._     _|_
  // |  (_| | _> (/_   _|_ | | |_) |_| |_
  //
  argparse::ArgumentParser program("4_buffer");

  program.add_argument("-g","--global")
   .help("Global Range")
   .default_value(1)
   .action([](const std::string& value) { return std::stoi(value); });

  try {
    program.parse_args(argc, argv);
  }
  catch (const std::runtime_error& err) {
    std::cout << err.what() << std::endl;
    std::cout << program;
    exit(0);
  }

  const auto global_range = program.get<int>("-g");

  //  _       _   _
  // |_)    _|_ _|_ _  ._
  // |_) |_| |   | (/_ |
  //

  // Create arrays
  std::vector<int> B(global_range);
  for (int i=0; i<global_range; i++) {
      B[i] = i;
  }
  std::vector<int> A(global_range);
  
  {
    // Create sycl buffer.
    // The buffer need to be destructed at the end of the scope to triger 
    // syncronization
    // Trivia: What happend if we create the buffer in the outer scope?
    sycl::buffer<sycl::cl_int, 1> bufferA(A.data(), A.size());
    sycl::buffer<sycl::cl_int, 1> bufferB(B.data(), B.size());
    
    sycl::queue myQueue;
    std::cout << "Running on "
              << myQueue.get_device().get_info<sycl::info::device::name>()
              << "\n";

    // Create a command_group to issue command to the group
    myQueue.submit([&](sycl::handler &cgh) {
      // Create an accesor for the sycl buffer. Trust me, use auto.
      auto accessorA = bufferA.get_access<sycl::access::mode::write>(cgh);
      auto accessorB = bufferB.get_access<sycl::access::mode::read>(cgh);
      // Submit the kernel
      cgh.parallel_for<class hello_world>(
          sycl::range<1>(global_range), 
          [=](sycl::id<1> idx) {
            // Use the accesor
            // id have some 'usefull' overwrite
            accessorA[idx] = accessorB[idx];
          }); // End of the kernel function
    });       // End of the queue commands
  }           // End of scope. Buffer destructor is blocking.

  for (size_t i = 0; i < global_range; i++) {
    std::cout << "A[ " << i << " ] = " << A[i] << " "
              << "B[ " << i << " ] = " << B[i] << std::endl;
  }
  return 0;
}
