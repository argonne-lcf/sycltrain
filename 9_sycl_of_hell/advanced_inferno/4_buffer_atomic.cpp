#include "argparse.hpp"
#include <CL/sycl.hpp>
#include <vector>

namespace sycl = cl::sycl;

// Note: please don't use std::atomic_ref<T> in the device-code
template <typename T>
using relaxed_atomic_ref =
    sycl::ONEAPI::atomic_ref< T,
                             sycl::ONEAPI::memory_order::relaxed,
                             sycl::ONEAPI::memory_scope::device,
                             sycl::access::address_space::global_space>;

int main(int argc, char **argv) {

  //  _                ___
  // |_) _. ._ _  _     |  ._  ._     _|_
  // |  (_| | _> (/_   _|_ | | |_) |_| |_
  //
  argparse::ArgumentParser program("4_buffer_atomic.cpp");

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

  // Crrate array
  std::vector<int> A(1);
  std::vector<int> A_atom(1);

  // Selectors determine which device kernels will be dispatched to.
  sycl::default_selector selector;
  // Create your own or use `{cpu,gpu,accelerator}_selector`
  {
    // Create sycl buffer.
    sycl::buffer<sycl::cl_int, 1> bufferA(A.data(), A.size());
    sycl::buffer<sycl::cl_int, 1> bufferA_atom(A_atom.data(), A_atom.size());

    sycl::queue myQueue(selector);
    std::cout << "Running on "
              << myQueue.get_device().get_info<sycl::info::device::name>()
              << "\n";

    // Create a command_group to issue command to the group
    myQueue.submit([&](sycl::handler &cgh) {
      // Create an accesor for the sycl buffer. Trust me, use auto.
      auto accessorA = bufferA.get_access<sycl::access::mode::read_write>(cgh);
      auto accessorA_atom =  bufferA_atom.get_access<sycl::access::mode::atomic>(cgh);
      // Range allow use to access information
      cgh.parallel_for<class hello_world>(
          sycl::range<1>(global_range),
          [=](sycl::id<1> _) {
            accessorA[0] = accessorA[0] + 1;

            //accessorA_atom[0].fetch_add(1); // will be depricated by SYCL2020

            auto atm = relaxed_atomic_ref<sycl::cl_int>( (accessorA_atom.get_pointer())[0] );
            atm.fetch_add( 1 );

          }); // End of the kernel function
    });       // End of the queue commands
  }           // End of scope, wait for the queued work to stop.

  std::cout <<"Counter incrememented " << global_range << " time " << std::endl;
  std::cout << "Atomic Increment " << A_atom[0] << std::endl;
  std::cout << "Race condition Increment " << A[0] << std::endl;

  assert(A_atom[0] == global_range );
  return 0;
}
