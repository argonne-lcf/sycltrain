#include "cxxopts.hpp"
#include <CL/sycl.hpp>

namespace sycl = cl::sycl;

int main(int argc, char **argv) {

  //  _                ___
  // |_) _. ._ _  _     |  ._  ._     _|_
  // |  (_| | _> (/_   _|_ | | |_) |_| |_
  //                           |

  cxxopts::Options options("2_parallel_for",
                           " How to use 'parallel_for' and 'range' ");

  options.add_options()("h,help", "Print help")(
      "g,grange", "Global Range", cxxopts::value<int>()->default_value("1"));

  auto result = options.parse(argc, argv);

  if (result.count("help")) {
    std::cout << options.help({"", "Group"}) << std::endl;
    exit(0);
  }

  const auto global_range = result["grange"].as<int>();

  //  _                             _
  // |_) _. ._ ._ _. | | |  _  |   |_ _  ._
  // |  (_| |  | (_| | | | (/_ |   | (_) |

  // Selectors determine which device kernels will be dispatched to.
  sycl::default_selector selector;
  {

    sycl::queue myQueue(selector);
    std::cout << "Running on "
              << myQueue.get_device().get_info<sycl::info::device::name>()
              << "\n";

    // Create a command_group to issue command to the group
    myQueue.submit([&](sycl::handler &cgh) {
      sycl::stream cout(1024, 256, cgh);
      // #pragma omp parallel for
      // for(int idx[0]=0; idx[0]++; idx[0]< global_range)
      cgh.parallel_for<class hello_world>(
          sycl::range<1>(global_range), [=](sycl::id<1> idx) {
            cout << "Hello, World: World rank " << idx[0] << sycl::endl;
          }); // End of the kernel function
    });       // End of the queue commands.
  }           // End of scope.Queue destruction will be called. W
              // This will wait for the queued work to stop.
  return 0;
}
