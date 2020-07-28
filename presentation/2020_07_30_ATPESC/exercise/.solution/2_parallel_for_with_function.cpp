#include "argparse.hpp"
#include <CL/sycl.hpp>
#include <math.h>   

namespace sycl = cl::sycl;

void bar(sycl::stream s, const sycl::id<1> idx)
{
    int repetition = idx[0] + 1;
    while(repetition--){
        s << "*" ;
    }
    s << sycl::endl;
}

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
    cgh.parallel_for<class hello_world>(
        sycl::range<1>(global_range), [=](sycl::id<1> idx) {
          bar(sout, idx);
        }); // End of the kernel function
  });       // End of the queue commands.
  return 0;
}
