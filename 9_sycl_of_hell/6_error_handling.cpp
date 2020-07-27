#include <argparse.hpp>
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
  const auto local_range = program.get<int>("-l");

  // ._   _|        ._ _. ._   _   _
  // | | (_|        | (_| | | (_| (/_
  //           __              _|

  // Selectors determine which device kernels will be dispatched to.
  sycl::default_selector selector;
  // Create your own or use `{cpu,gpu,accelerator}_selector`

  // Create you async handler
  sycl::async_handler ah = [](sycl::exception_list elist) {
    for (auto e : elist)
      std::rethrow_exception(e);
  };

    sycl::queue myQueue(selector, ah);
    std::cout << "Running on "
              << myQueue.get_device().get_info<sycl::info::device::name>()
              << "\n";
   try {
    // Create a command_group to issue command to the group
    myQueue.submit([&](sycl::handler &cgh) {
      // Create a output stream (lot of display, lot of number)
      sycl::stream sout(10240, 2560, cgh);

      // nd_range, geneate a nd_item who allow use to query loop dispach
      // information
      cgh.parallel_for<class hello_world>(
          sycl::nd_range<1>{sycl::range<1>(global_range),
                            sycl::range<1>(local_range)},
          [=](sycl::nd_item<1> idx) {
            const int world_rank = idx.get_global_id(0);
            const int work_size = idx.get_global_range(0);
            const int local_rank = idx.get_local_id(0);
            const int local_size = idx.get_local_range(0);
            const int group_rank = idx.get_group(0);
            const int group_size = idx.get_group_range(0);

            sout << "Hello world: World rank/size: " << world_rank << " / "
                 << work_size << ". Local rank/size: " << local_rank << " / "
                 << local_size << ". Group rank/size: " << group_rank << " / "
                 << group_size << sycl::endl;
          }); // End of the kernel function
    });       // End of the queue commands
    myQueue.wait_and_throw();
  } catch (sycl::exception &e) {
    std::cout << "Async Exception: " << e.what() << std::endl;
  }

  return 0;
}
