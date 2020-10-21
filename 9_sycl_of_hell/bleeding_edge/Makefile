vadd_usm:
	dpcpp vadd_usm.cpp -fsycl-explicit-simd -o vadd_usm

run_vadd_usm:
	SYCL_PROGRAM_COMPILE_OPTIONS=-vc-codegen ./vadd_usm

omp_syc:
	icpx -fiopenmp -fopenmp-targets=spir64 -fsycl omp_sycl.cpp

run_comp_syc:
	./run_comp_syc


clean:
	rm -rf vadd_usm omp_syc
