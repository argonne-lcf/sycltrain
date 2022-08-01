#include <sycl/sycl.hpp>
// Inspired by Victor (vanisimov@anl.gov) example.

int main() {

  // Select a device
  sycl::queue  Q;
  std::cout << "Device Name: " << Q.get_device().get_info<sycl::info::device::name>() << std::endl;

  int *a = sycl::malloc_shared<int>(1, Q);
  a[0] = 0;

  auto e = Q.single_task( [=]() { a[0] = 1; } );

  Q.submit([&](sycl::handler &cgh) {
      cgh.depends_on(e);
      cgh.host_task( [=]() {
         std::cout << "Host Task a[0] " <<  a[0] << std::endl;
         std::cout << "Expected 1" << std::endl;
         assert(a[0] == 1);
      });
  }).wait();
}
