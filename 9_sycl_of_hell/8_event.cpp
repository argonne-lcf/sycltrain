#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

int main() {
  // If one use event to use profiling, you need to enable your queue for that
  sycl::queue Q{sycl::property::queue::enable_profiling()};
  std::cout << "Running on "
            << Q.get_device().get_info<sycl::info::device::name>()
            << "\n";
  //  _               
  // |_     _  ._ _|_ 
  // |_ \/ (/_ | | |_ 
  //                  

  // Event are used for 2 things:
  //   - Ordering and
  //   - Timing

  // The queue submition return an event,
  // That we can use for synchronizing kernel submision, or like in this example,
  // or gather proffiling information
  cl::sycl::event e0 = Q.submit([&](sycl::handler &cgh) {
    sycl::stream sout(1024, 256, cgh);
    cgh.single_task<class hello_world>([=]() {
       sout << "Hello, World 0!" << sycl::endl;
    }); 
  }); 

  cl::sycl::event e1 = Q.submit([&](sycl::handler &cgh) {
    //This kernel will wait that e0 finish before being submited
    // Without it, because Queue are out-of-order by default, the order was non deterministic
    cgh.depends_on(e0);
    sycl::stream sout(1024, 256, cgh);
    cgh.single_task<class hello_world>([=]() {
       sout << "Hello, World 1!" << sycl::endl;
    });
  });

  // We want to gather information on the execution time of the kernel
  // But At this point in time we don't know if the kernel is finished or not.
  // Fortunaly,  `get_profiling_info` will wait for the event to be completed
  // using implicit the `wait_for` sycl function.  
  auto ns =  e1.get_profiling_info<sycl::info::event_profiling::command_end>()-e0.get_profiling_info<sycl::info::event_profiling::command_start>();
  std::cout <<  "Both kernels tooks " << ns  << " ns" << std::endl;
  return 0;
}
