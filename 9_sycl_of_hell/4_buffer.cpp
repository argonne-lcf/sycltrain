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
  std::vector<int> A(global_range);
  {
    // Create sycl buffer.
    // The buffer need to be destructed at the end of the scope to trigger 
    // synchronization
    // Trivia: What happend if we create the buffer in the outer scope?
    sycl::buffer bufferA{A};
    // In case of raw pointer one should use
    // sycl::buffer<sycl::cl_int,1> bufferA(A.data(), A.size());

    sycl::queue Q;
    std::cout << "Running on "
              << Q.get_device().get_info<sycl::info::device::name>()
              << "\n";

    Q.submit([&](sycl::handler &cgh) {
      // Create an accesor for the sycl buffer
      sycl::accessor accessorA{bufferA, cgh, sycl::write_only, sycl::noinit};
      // Submit the kernel
      cgh.parallel_for(
          sycl::range<1>(global_range), 
          [=](sycl::id<1> idx) {
            // Use the accesor
            // id<1> have some 'usefull' overwrite
            accessorA[idx] = idx;
          });
    });    
  }// End of the buffer scope, wait for the queued work to stop.

  for (size_t i = 0; i < global_range; i++)
    std::cout << "A[ " << i << " ] = " << A[i] << std::endl;
  return 0;
}
