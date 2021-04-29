#include "argparse.hpp"
#include <CL/sycl.hpp>

namespace sycl = cl::sycl;

int main(int argc, char **argv) {

  //  _                ___
  // |_) _. ._ _  _     |  ._  ._     _|_
  // |  (_| | _> (/_   _|_ | | |_) |_| |_
  //                           |

  argparse::ArgumentParser program("5_buffer_usm");

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

 //            _           __                                              
 // | | ._  o _|_ _   _|   (_  |_   _. ._ _   _|   ._ _   _  ._ _   _  ._   
 // |_| | | |  | (/_ (_|   __) | | (_| | (/_ (_|   | | | (/_ | | | (_) | \/ 
 //                                                                      / 

  sycl::queue Q;

  int *A = sycl::malloc_shared<int>(global_range, Q);
  // Advise runtime how memory will be used
  //auto e = myQueue.mem_advise(A, global_range * sizeof(int), PI_MEM_ADVICE_SET_NON_ATOMIC_MOSTLY);
  //e.wait();

  std::cout << "Running on "
            << Q.get_device().get_info<sycl::info::device::name>()
            << "\n";

  // Create a command_group to issue command to the group
  Q.parallel_for(global_range, [=](sycl::item<1> id) { A[id] = id; }).wait();

  for (size_t i = 0; i < global_range; i++)
    std::cout << "A[ " << i << " ] = " << A[i] << std::endl;
  return 0;
}
