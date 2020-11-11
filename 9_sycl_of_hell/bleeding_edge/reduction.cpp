#include <CL/sycl.hpp>
#include <vector>

namespace sycl = cl::sycl;

int main(int argc, char **argv) {
  const auto global_range = 10;

  sycl::queue myQueue;
  std::cout << "Running on "
              << myQueue.get_device().get_info<sycl::info::device::name>()
              << "\n";

  // Crrate array
  std::vector<float> A(global_range);
  A[0] = 1.0;
  // If one want to play with USM 
  //sycl::device dev = myQueue.get_device();
  //sycl::context ctex = myQueue.get_context();
  //float* sum = static_cast<float*>(sycl::malloc_shared(1 * sizeof(float), dev, ctex));
  float sum;
  {
    sycl::buffer<sycl::cl_float, 1> bufferA(A.data(), A.size());
    sycl::buffer<sycl::cl_float, 1> bufferS(&sum, 1);

    myQueue.submit([&](sycl::handler &cgh) {
      auto accessorA = bufferA.get_access<sycl::access::mode::read>(cgh);
      //See 4.10.2 of SYCL 2020 provitional spec for me information
      auto accessorS = bufferS.get_access<sycl::access::mode::write, sycl::access::target::global_buffer>(cgh);
      
      cgh.parallel_for(
          sycl::nd_range<1>(global_range,1),
          // `identity` not yet implemented
          sycl::ONEAPI::reduction(accessorS, std::plus<float>()), 
          [=](sycl::nd_item<1> it, auto& accessorS) {
            const int i = it.get_global_id(0);  
            accessorS += accessorA[i];
          }); 
    });      
  }         
  std::cout << "sum:" << sum << std::endl;
  return 0;
}
