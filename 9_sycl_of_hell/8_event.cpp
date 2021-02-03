#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

int main() {
  sycl::queue myQueue{sycl::property::queue::enable_profiling()};
  std::cout << "Running on "
            << myQueue.get_device().get_info<sycl::info::device::name>()
            << "\n";
    //  _               
    // |_     _  ._ _|_ 
    // |_ \/ (/_ | | |_ 
    //                  

    // The queue submition return an event,
    // That we can use for synchronizing kernel submision, or like in this example,
    // Gather proffiling information
    cl::sycl::event e = myQueue.submit([&](sycl::handler &cgh) {
      sycl::stream sout(1024, 256, cgh);

      cgh.single_task<class hello_world>([=]() {
        sout << "Hello, World!" << sycl::endl;
      }); 
    }); 

    // wait for all queue submissions to complete
    myQueue.wait();
  
  // We want to gather information on the execution time of the kernel
  // But At this point in time we don't know if the kernel is finished or not.
  // Fortunaly,  `get_profiling_info` will wait for the event to be completed
  // using implicit the `wait_for` sycl function.  
  auto ns =  e.get_profiling_info<sycl::info::event_profiling::command_end>() -e.get_profiling_info<sycl::info::event_profiling::command_start>();
  std::cout <<  "This kernel took " << ns  << " ns" << std::endl;
  return 0;
}
