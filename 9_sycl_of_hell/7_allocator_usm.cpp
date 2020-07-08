#include "cxxopts.hpp"
#include <CL/sycl.hpp>
#include <vector>

namespace sycl = cl::sycl;

int main(int argc, char **argv) {

  //  _                ___
  // |_) _. ._ _  _     |  ._  ._     _|_
  // |  (_| | _> (/_   _|_ | | |_) |_| |_
  //

  cxxopts::Options options("7_allocator_usm", "Using allocator");

  options.add_options()("h,help", "Print help")(
      "g,grange", "Global Range", cxxopts::value<int>()->default_value("1"));

  auto result = options.parse(argc, argv);

  if (result.count("help")) {
    std::cout << options.help({"", "Group"}) << std::endl;
    exit(0);
  }

  const auto global_range = result["grange"].as<int>();

  // Selectors determine which device kernels will be dispatched to.
  sycl::default_selector selector;
  sycl::queue myQueue(selector);

  // Create usm allocator
  sycl::usm_allocator<float, sycl::usm::alloc::shared> allocator(
      myQueue.get_context(), myQueue.get_device());
  // Allocate value
  std::vector<float, decltype(allocator)> A(global_range, allocator);

  std::cout << "Running on "
            << myQueue.get_device().get_info<sycl::info::device::name>()
            << "\n";

// A vector is not trivialy copyable
  auto *A_p = A.data();

  // Create a command_group to issue command to the group
  myQueue.submit([&](sycl::handler &cgh) {
    // No accessor needed!
    cgh.parallel_for<class hello_world>(
        sycl::range<1>{sycl::range<1>(global_range)},
        [=](sycl::id<1> idx_id) {
	  const int idx = idx_id[0];
          A_p[idx] = idx;
        }); // End of the kernel function
  });       // End of the queue commands
  myQueue.wait();

  for (size_t i = 0; i < global_range; i++)
    std::cout << "A[ " << i << " ] = " << A[i] << std::endl;
  return 0;
}
