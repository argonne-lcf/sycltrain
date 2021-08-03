#include <CL/sycl.hpp>
#include <numeric>
namespace sycl = cl::sycl;

template <typename T>
T foo(sycl::queue Q, int N){
 T *A = sycl::malloc_shared<T>(N,Q);
 Q.parallel_for(N, [=](sycl::item<1> id) { A[id] = id; }).wait();
 return std::accumulate(A, A+N, T{});
}


int main() {
 sycl::queue Q{sycl::gpu_selector{}};
 std::cout << "Running on "
            << Q.get_device().get_info<sycl::info::device::name>()
            << std::endl;
 const int N{1729};
 assert (foo<int>(Q,N) == N*(N-1)/2); 
}
