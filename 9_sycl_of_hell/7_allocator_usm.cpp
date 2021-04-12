#include "argparse.hpp"
#include <CL/sycl.hpp>
#include <vector>

namespace sycl = cl::sycl;

int main(int argc, char **argv) {

  //  _                ___
  // |_) _. ._ _  _     |  ._  ._     _|_
  // |  (_| | _> (/_   _|_ | | |_) |_| |_
  //
  argparse::ArgumentParser program("7_allocator_usm");

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

  sycl::queue myQueue;

  // Create usm allocator
  sycl::usm_allocator<float, sycl::usm::alloc::shared> allocator(myQueue);
  // Allocate value
  std::vector<float, decltype(allocator)> A(global_range, allocator);

  std::cout << "Running on "
            << myQueue.get_device().get_info<sycl::info::device::name>()
            << "\n";

  // A vector is not trivialy copyable
  auto *A_p = A.data();

  // Create a command_group to issue command to the group
  myQueue.parallel_for<class hello_world>(
        sycl::range<1>{sycl::range<1>(global_range)},
        [=](sycl::id<1> idx_id) {
          const int idx = idx_id[0];
          A_p[idx] = idx;
  }).wait();

  for (size_t i = 0; i < global_range; i++)
    std::cout << "A[ " << i << " ] = " << A[i] << std::endl;
  return 0;
}
