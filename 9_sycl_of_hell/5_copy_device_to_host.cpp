#include "argparse.hpp"
#include <CL/sycl.hpp>

namespace sycl = cl::sycl;

int main(int argc, char **argv) {

  //  _                ___
  // |_) _. ._ _  _     |  ._  ._     _|_
  // |  (_| | _> (/_   _|_ | | |_) |_| |_
  //                           |
  argparse::ArgumentParser program("5_copy_device_to_host");

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
  // The buffer is created outside of the scope
  sycl::buffer bufferA(A);

  sycl::queue Q;
  std::cout << "Running on "
            << Q.get_device().get_info<sycl::info::device::name>()
            << "\n";
 
  // Create a command_group to issue command to the group
  Q.submit([&](sycl::handler &cgh) {
      sycl::accessor accessorA{bufferA, cgh, sycl::write_only, sycl::noinit};
      cgh.parallel_for(
          sycl::range<1>(global_range),
          [=](sycl::id<1> idx) {
            accessorA[idx] = idx;
      });
  }); // SYCL Queue are by default out-of-order
  // But accessors will handle the dependency dag for you

  // Now update the host buffer 
  Q.submit([&](sycl::handler &cgh) {
      sycl::accessor accessorA{bufferA, cgh, sycl::read_only};
      // If you can prove that `bufferA` have no internal copy
      cgh.update_host(accessorA);
      // else one can use the more general
      // cgh.copy(accessorA,A.data());
  });
  // The synchronization append at the buffer destructor,
  // buffer is at the global scope so we need to explicitly wait
  Q.wait();

  for (size_t i = 0; i < global_range; i++)
    std::cout << "A[ " << i << " ] = " << A[i] << std::endl;
  return 0;
}
