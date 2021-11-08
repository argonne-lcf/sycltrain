#include "argparse.hpp"
#include <sycl/sycl.hpp>



int main(int argc, char **argv) {

  //  _                ___
  // |_) _. ._ _  _     |  ._  ._     _|_
  // |  (_| | _> (/_   _|_ | | |_) |_| |_
  //                           |
  argparse::ArgumentParser program("2_parallel_for");

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
  //  _                          _
  // |_) _. ._ _. | |  _  |   |_ _  ._
  // |  (_| | (_| | | (/_ |   | (_) |

  sycl::queue Q;
  std::cout << "Running on "
            << Q.get_device().get_info<sycl::info::device::name>()
            << std::endl;

  // Create a command_group to issue command to the group
  Q.submit([&](sycl::handler &cgh) {
    sycl::stream sout(1024, 256, cgh);
    // #pragma omp parallel for
    cgh.parallel_for(
        // for(int idx=0; idx++; idx < global_range)
        global_range, [=](sycl::id<1> idx) {
          sout << "Hello, World: World rank " << idx << sycl::endl;
        }); // End of the kernel function
  }).wait();       // End of the queue commands we waint on the event reported.
  //One can also wait for all the job on the Queue to be finished
  //Q.wait();

  return 0;
}
