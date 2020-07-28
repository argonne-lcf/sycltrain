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

  // Create array
  std::vector<int> A(global_range);
  std::vector<int> B(global_range);
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
      auto accessorB = bufferB.get_access<sycl::access::mode::discard_write>(cgh);
      // Submit the kernel
      cgh.parallel_for<class hello_world>(
          sycl::range<1>(global_range), 
          [=](sycl::id<1> idx) {
            accessorB[idx] = idx[0];
          }); // End of the kernel function
    });       // End of the queue commands

    // THis kernel will read from accessorB.
    // It will wait for the previous kernel to finish before beeing scheduled
    myQueue.submit([&](sycl::handler &cgh) {
      auto accessorA = bufferA.get_access<sycl::access::mode::write>(cgh);
      auto accessorB = bufferB.get_access<sycl::access::mode::read>(cgh);
      // Submit the kernel
      cgh.parallel_for<class hello_world>(
          sycl::range<1>(global_range),
          [=](sycl::id<1> idx) {
            accessorA[idx] = accessorB[idx];
          }); 
    });      
  }

  for (size_t i = 0; i < global_range; i++) {
    std::cout << "A[ " << i << " ] = " << A[i] << " "
              << "B[ " << i << " ] = " << B[i] << std::endl;
  }

  return 0;
}
