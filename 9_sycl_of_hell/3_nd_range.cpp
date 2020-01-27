#include <CL/sycl.hpp>

// Inspired by Codeplay compute cpp hello-world
int main(int argc, char** argv) {
  const auto global_range =  (size_t) atoi(argv[1]);
  const auto local_range =  (size_t) atoi(argv[2]);
  
  // Selectors determine which device kernels will be dispatched to.
  cl::sycl::default_selector selector; 
  // Create your own or use `{cpu,gpu,accelerator}_selector`
  {
  
  cl::sycl::queue myQueue(selector);
  std::cout << "Running on "
            << myQueue.get_device().get_info<cl::sycl::info::device::name>()
            << "\n";

// ._   _|        ._ _. ._   _   _
// | | (_|        | (_| | | (_| (/_
//           __              _|


  //Create a command_group to issue command to the group
  myQueue.submit([&](cl::sycl::handler& cgh) {

    // Create a output stream (lot of display, lot of number)
    cl::sycl::stream cout(10240, 2560, cgh);

    // nd_range, geneate a nd_item who allow use to query loop dispach information
    cgh.parallel_for<class hello_world>(cl::sycl::nd_range<1>{cl::sycl::range<1>(global_range), 
                                                             cl::sycl::range<1>(local_range) }, 
                                        [=](cl::sycl::nd_item<1> idx) {
        const int world_rank = idx.get_global_id(0);
        const int work_size = idx.get_global_range(0);
        const int local_rank = idx.get_local_id(0);
        const int local_size = idx.get_local_range(0);
        const int group_rank = idx.get_group(0);
        const int group_size = idx.get_group_range(0); 


       cout<< "Hello world: World rank/size: " <<  world_rank << " / " << work_size  << 
              ". Local rank/size: "             << local_rank << " / " << local_size << 
              ". Group rank/size: "             << group_rank << " / " << group_size << 
              cl::sycl::endl;

    }); // End of the kernel function
  }); // End of the queue commands 
  }  // End of scope, wait for the queued work to stop. 
     // Can also use  myQueue.wait_and_throw();
  return 0;
}
