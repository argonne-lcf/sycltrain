#include <CL/sycl.hpp>

int main(int argc, char** argv) {
  const auto global_range =  (size_t) atoi(argv[1]);

  // Selectors determine which device kernels will be dispatched to.
  cl::sycl::default_selector selector; 
  // Create your own or use `{cpu,gpu,accelerator}_selector`
  {
  
  cl::sycl::queue myQueue(selector);
  std::cout << "Running on "
            << myQueue.get_device().get_info<cl::sycl::info::device::name>()
            << "\n";
//  _                             _       
// |_) _. ._ ._ _. | | |  _  |   |_ _  ._ 
// |  (_| |  | (_| | | | (/_ |   | (_) |                                          
                        
  //Create a command_group to issue command to the group
  myQueue.submit([&](cl::sycl::handler& cgh) {

    // Create a output stream
    cl::sycl::stream cout(1024, 256, cgh);
    //
    // #pragma omp parallel for
    // for(int idx[0]=0; idx[0]++; idx[0]< global_range)
    cgh.parallel_for<class hello_world>(cl::sycl::range<1>(global_range), 
                                        [=](cl::sycl::id<1> idx) {
       cout << "Hello, World: World rank " << idx[0] << cl::sycl::endl;
    }); // End of the kernel function
  }); // End of the queue commands 
  }  // End of scope, wait for the queued work to stop. 
     // Can also use  myQueue.wait_and_throw();
  return 0;
}
