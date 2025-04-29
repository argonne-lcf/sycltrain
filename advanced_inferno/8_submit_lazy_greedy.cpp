#include <chrono>
#include <iostream>
#include <sycl/sycl.hpp>
#include <thread>

int main(int argc, char **argv) {
  sycl::queue Q;
  auto *ptr = sycl::malloc_shared<int>(1, Q);
  auto f = [=]() {
    sycl::atomic_ref<int, sycl::memory_order_acq_rel,
                     sycl::memory_scope::system>(*ptr)
        .store(1);
  };
  
  *ptr = 0;
  const auto start = std::chrono::high_resolution_clock::now();
  Q.single_task(f);
  Q.wait();
  const auto end = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::micro> elapsed = (end - start);

  *ptr = 0;
  std::cout << "Submitting Kernel who will set the sentinel to 1" << std::endl;
  Q.single_task(f);
  std::cout << "Sleep 4 times the estimated durations of the kernel"
            << std::endl;
  std::this_thread::sleep_for(elapsed * 4);
  std::cout << "Setting the sentinel to 2" << std::endl;
  *ptr = 2;
  std::cout << "Waiting for the Kernel to finish" << std::endl;
  Q.wait();

  std::cout << "Sentinel Value: " << *ptr << std::endl;
  if (*ptr == 1)
    std::cout << "Lazy Execution" << std::endl;
  else
    std::cout << "Greedy Execution" << std::endl;
}
