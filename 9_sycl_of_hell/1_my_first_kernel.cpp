#include <CL/sycl.hpp>

// Inspired by Codeplay compute cpp hello-world
int main() {
  // Selectors determine which device kernels will be dispatched to.
  cl::sycl::default_selector selector; 
  // Create your own or use `{cpu,gpu,accelerator}_selector`
  {
  
  cl::sycl::queue myQueue(selector);
  std::cout << "Running on "
            << myQueue.get_device().get_info<cl::sycl::info::device::name>()
            << "\n";
//         __                     ___          
//  /\    (_  o ._ _  ._  |  _     |  _.  _ |  
// /--\   __) | | | | |_) | (/_    | (_| _> |< 
//                    |                        
                                    
  //Create a command_group to issue command to the group
  myQueue.submit([&](cl::sycl::handler& cgh) {
    // Create a output stream
    cl::sycl::stream cout(1024, 256, cgh);

    cgh.single_task<class hello_world>([=]() {
    cout << "Hello, World!" << cl::sycl::endl;
    }); // End of the kernel function
  }); // End of the queue commands 
  }  // End of scope, wait for the queued work to stop. 
     // Can also use  myQueue.wait_and_throw();
  return 0;
}
