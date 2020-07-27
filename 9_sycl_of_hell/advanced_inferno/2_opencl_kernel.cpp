#include <CL/sycl.hpp>

// Maybe in the futur sycl will note be in the 'cl' namespace
namespace sycl = cl::sycl;

int main() {
  // Selectors determine which device kernels will be dispatched to.
  // Create your own or use `{cpu,gpu,accelerator}_selector`
  sycl::default_selector selector;

  // SYCL rely heavily on constructor / destructor semantics
  // e.g the destructor of the queue, will wait for all the kernel
  // submited to this queue to terminate
  {

    sycl::queue myQueue(selector);
    std::cout << "Running on "
              << myQueue.get_device().get_info<sycl::info::device::name>()
              << "\n";

    sycl::program p(myQueue.get_context());
    // build_with_source may take an aditioanl argument to pass compile flags
    p.build_with_source(R"EOL(__kernel void hello_world() {printf("Hello world\n");} )EOL");
        
    //  _               _                          
    // / \ ._   _  ._  /  |    |/  _  ._ ._   _  | 
    // \_/ |_) (/_ | | \_ |_   |\ (/_ |  | | (/_ | 
    //     |                                      
    // Create a command_group to issue command to the group.
    myQueue.submit([&](sycl::handler &cgh) {
      // Will launch an opencl kernel
      // Use cgh.set_args($acceors) if required by your kermel
      cgh.single_task(p.get_kernel("hello_world"));
    });
  } // End of scopes, the queue will be destroyed, trigering a synchronization
  return 0;
}
