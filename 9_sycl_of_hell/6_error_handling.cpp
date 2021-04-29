#include "argparse.hpp"
#include <CL/sycl.hpp>

namespace sycl = cl::sycl;

int main(int argc, char **argv) {

  //  _                ___
  // |_) _. ._ _  _     |  ._  ._     _|_
  // |  (_| | _> (/_   _|_ | | |_) |_| |_
  //
  argparse::ArgumentParser program("6_error_handling");

  program.add_argument("-g","--global")
   .help("Global Range")
   .default_value(1)
   .action([](const std::string& value) { return std::stoi(value); });

  program.add_argument("-l","--local")
   .help("Local Range")
   .default_value(2)
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
  const auto local_range = program.get<int>("-l");

  // ._   _|        ._ _. ._   _   _
  // | | (_|        | (_| | | (_| (/_
  //           __              _|

  // Selectors determine which device kernels will be dispatched to.
  sycl::default_selector selector;
  // Create you async handler
  sycl::async_handler ah = [](sycl::exception_list elist) {
    for (auto e : elist)
      std::rethrow_exception(e);
  };

  sycl::queue Q(selector, ah);
  std::cout << "Running on "
            << Q.get_device().get_info<sycl::info::device::name>()
            << "\n";
  try {
  // Create a command_group to issue command to the group
  Q.submit([&](sycl::handler &cgh) {
    sycl::stream sout(10240, 2560, cgh);
    cgh.parallel_for(
        sycl::nd_range<1>{sycl::range<1>(global_range),
                          sycl::range<1>(local_range)},
          [=](sycl::nd_item<1> idx) {
            sout << "Hello world: World rank/size: " << idx.get_global_id(0) <<  sycl::endl;
          }); // End of the kernel function
    }).wait_and_throw();       // End of the queue commands  
   } catch (sycl::exception &e) {
    std::cout << "Async Exception: " << e.what() << std::endl;
  }

  return 0;
}
