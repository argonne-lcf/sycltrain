#include "cxxopts.hpp"
#include <CL/sycl.hpp>

namespace sycl = cl::sycl;

int main(int argc, char **argv) {

  //  _                ___
  // |_) _. ._ _  _     |  ._  ._     _|_
  // |  (_| | _> (/_   _|_ | | |_) |_| |_
  //

  cxxopts::Options options("3_nd_range", " How to use 'nd_range' ");

  options.add_options()("h,help", "Print help")(
      "g,grange", "Global Range", cxxopts::value<int>()->default_value("1"))(
      "l,lrange", "Local Range", cxxopts::value<int>()->default_value("1"))

      ;

  auto result = options.parse(argc, argv);

  if (result.count("help")) {
    std::cout << options.help({"", "Group"}) << std::endl;
    exit(0);
  }

  const auto global_range = result["grange"].as<int>();
  const auto local_range = result["lrange"].as<int>();

  // ._   _|        ._ _. ._   _   _
  // | | (_|        | (_| | | (_| (/_
  //           __              _|

  // Selectors determine which device kernels will be dispatched to.
  sycl::default_selector selector;
  // Create your own or use `{cpu,gpu,accelerator}_selector`
  {

    sycl::queue myQueue(selector);
    std::cout << "Running on "
              << myQueue.get_device().get_info<sycl::info::device::name>()
              << "\n";

    // Create a command_group to issue command to the group
    myQueue.submit([&](sycl::handler &cgh) {
      // Create a output stream (lot of display, lot of number)
      sycl::stream cout(10240, 2560, cgh);

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

            cout << "Hello world: World rank/size: " << world_rank << " / "
                 << work_size << ". Local rank/size: " << local_rank << " / "
                 << local_size << ". Group rank/size: " << group_rank << " / "
                 << group_size << sycl::endl;
          }); // End of the kernel function
    });       // End of the queue commands
  }           // End of scope, destructor, wait for the queued work to stop.
              // Can also use  myQueue.wait_and_throw();
  return 0;
}
