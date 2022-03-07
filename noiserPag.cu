// noiserPag.cu
// Creation 20210205
// by Mert SIDE

#include <stdio.h>
#include <assert.h>

#include <stdlib.h>
#include <time.h>

#include <unistd.h>

#include <x86intrin.h>
#ifdef    NO_LFENCE
#define   lfence()
#else
#include <emmintrin.h>
#define   lfence()  _mm_lfence()
#endif

#define TIMES_MEASURED 100000
#define TIMES_REPEAT 10

// Compiled assembly for Intel:
//                                objdump sender.out -d | less
// Compiled assembly for Nvidia:
//                                cuobjdump -ptx sender.out | less
// https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
//  [HOST]  ---  ---  ---  ---  checkCuda  ---  ---  ---  ---  [by NVIDIA]
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

//  [HOST]  ---  ---  ---  ---  myRandom  ---  ---  ---  ---  [by meside]
unsigned int myRandom(unsigned int lower, unsigned int upper) 
{
  return (rand() % (upper - lower + 1)) + lower;
}

//  [HOST]  ---  ---  ---  getPageableH2DBandwidth  ---  ---  ---  [by meside]
void getPageableH2DBandwidth() 
{
  // Pinned transfers Host to Device bandwidth (GB/s):
  unsigned int nElements = 8*1024*1024;                         // 4MB
  const unsigned int bytes = nElements * sizeof(float);         // 16MB
  printf("bytes: %u\n", bytes);

  // host array
  float *host_arr;

  // device array
  float *dev_arr;

  // allocate and initialize
  host_arr = (float*)malloc(bytes);                       // host pageable
  checkCuda( cudaMalloc((void**)&dev_arr, bytes) );       // device
  memset(host_arr, 0, bytes);

  // initialize random number generator
  srand(time(0));

  for(int i = 0; i < TIMES_MEASURED; ++i) {
    int randNum = myRandom(0, 9);
    int randSiz = myRandom(sizeof(float), bytes);
    printf("%d\n", randNum);
    if(randNum < 1) {
      printf("    %d: %u\n", randNum, randSiz);
      checkCuda( cudaMemcpy(dev_arr, host_arr, randSiz, cudaMemcpyHostToDevice) );
    }
    cudaDeviceSynchronize();
    usleep(500);
  }

  // cleanup
  cudaFree(dev_arr);
  cudaFreeHost(host_arr);
}

//  [HOST]  ---  ---  ---  ---  ---  ---  main  ---  ---  ---  ---  ---  ---
int main()
{
  printf("\nTIMES_MEASURED: %d | TIMES_REPEAT: %d \n", 
  TIMES_MEASURED, TIMES_REPEAT);

  printf("getPageableH2DBandwidth:\n");
  getPageableH2DBandwidth();
  return 0;
}