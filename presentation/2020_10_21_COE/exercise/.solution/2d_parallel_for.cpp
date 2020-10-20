#include "argparse.hpp"
#include <CL/sycl.hpp>

namespace sycl = cl::sycl;

int main(int argc, char **argv) {

  //  _                ___
  // |_) _. ._ _  _     |  ._  ._     _|_
  // |  (_| | _> (/_   _|_ | | |_) |_| |_
  //                           |
  argparse::ArgumentParser program("2_parallel_for");

  program.add_argument("-g1","--global1")
   .help("Global Range first dimension")
   .default_value(1)
   .action([](const std::string& value) { return std::stoi(value); });
  program.add_argument("-g2","--global2")
   .help("Global Range second dimension")
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

  const auto global_range1 = program.get<int>("-g1");
  const auto global_range2 = program.get<int>("-g2");
  //  _                             _
  // |_) _. ._ ._ _. | | |  _  |   |_ _  ._
  // |  (_| |  | (_| | | | (/_ |   | (_) |

  sycl::queue myQueue;
  std::cout << "Running on "
            << myQueue.get_device().get_info<sycl::info::device::name>()
            << std::endl;
  // Create a command_group to issue command to the group
  myQueue.submit([&](sycl::handler &cgh) {
    sycl::stream sout(1024, 256, cgh);
    // #pragma omp parallel for
    cgh.parallel_for<class hello_world>(
        // for(int idx=0; idx++; idx < global_range)
        sycl::range<2>(global_range1,global_range2), [=](sycl::id<2> idx) {
          sout << "Hello, World: World rank " << idx << sycl::endl;
        }); // End of the kernel function
  });       // End of the queue commands.
  return 0;
}
