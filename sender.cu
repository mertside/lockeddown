// sender.cu
// Creation 20210209
// by Mert SIDE

#include <stdio.h>
#include <assert.h>

#include <unistd.h> // usleep

#include <time.h> //nanosleep

#include <x86intrin.h>
#ifdef    NO_LFENCE
#define   lfence()
#else
#include <emmintrin.h>
#define   lfence()  _mm_lfence()
#endif

#define TIMES_MEASURED 1000
#define TIMES_REPEAT 1
//#define NOP_DELAY 10 // for 8MB // 50 // for 16MB
// #define NOP_REPEAT 5000 // 100000 for 14 zeros
#define NOP_REPEAT 5500

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

//  [HOST]  ---  ---  ---  infiniteCpyPin  ---  ---  ---  ---  ---  [by meside]
void infiniteCpyPin(const unsigned int bytes) 
{
  // Pinned transfers Host to Device bandwidth (GB/s):
  //printf("bytes: %u\n", bytes);

  // host array
  float *host_arr;

  // device array
  float *dev_arr;

  // allocate and initialize
  checkCuda( cudaMallocHost((void**)&host_arr, bytes) );  // host pinned
  checkCuda( cudaMalloc((void**)&dev_arr, bytes) );       // device
  memset(host_arr, 0, bytes);

  // perform copies and report bandwidth
  while (1) {
    // BEWARE you are not checking for errors! see cudaError_t checkCuda(
    cudaMemcpy(dev_arr, host_arr, bytes, cudaMemcpyHostToDevice);
  }

  // cleanup
  cudaFree(dev_arr);
  cudaFreeHost(host_arr);
}

//  [HOST]  ---  ---  ---  infiniteCpyPag  ---  ---  ---  ---  ---  [by meside]
void infiniteCpyPag(const unsigned int bytes) 
{
  // Pinned transfers Host to Device bandwidth (GB/s):
  //printf("bytes: %u\n", bytes);

  // host array
  float *host_arr;

  // device array
  float *dev_arr;

  // allocate and initialize
  host_arr = (float*)malloc(bytes);                       // host pageable
  checkCuda( cudaMalloc((void**)&dev_arr, bytes) );       // device
  memset(host_arr, 0, bytes);

  // perform copies and report bandwidth
  while (1) {
    // BEWARE you are not checking for errors! see cudaError_t checkCuda(
    cudaMemcpy(dev_arr, host_arr, bytes, cudaMemcpyHostToDevice);
  }

  // cleanup
  free(dev_arr);
  cudaFreeHost(host_arr);
}

//  [HOST]  ---  ---  ---  latencyPerSizePin  ---  ---  ---  [by meside]
void latencyPerSizePin(const unsigned int bytes) 
{
  // Pinned transfers Host to Device bandwidth (GB/s):

  // host array
  float *host_arr;
  // device array
  float *dev_arr;

  // allocate and initialize
  checkCuda( cudaMallocHost((void**)&host_arr, bytes) );  // host pinned
  checkCuda( cudaMalloc((void**)&dev_arr, bytes) );       // device
  memset(host_arr, 0, bytes);

  unsigned long long time[TIMES_MEASURED];
  unsigned long long startTime, endTime;
  // perform copies and report bandwidth
  for(int i = 0; i < TIMES_MEASURED; ++i) {
    // First, let's ensure our CPU executes everything thus far
    lfence();
    // Start timing
    startTime = __rdtsc();
    // Ensure timing starts before we call the function
    lfence();
    
    for(int j = 0; j < TIMES_REPEAT; ++j) {
      // BEWARE you are not checking for errors! see cudaError_t checkCuda()
      cudaMemcpy(dev_arr, host_arr, bytes, cudaMemcpyHostToDevice);
    }

    // Ensure everything has been executed thus far
    lfence();
    // Stop timing.
    endTime = __rdtsc();
    // Ensure we have the counter value before proceeding 
    lfence();
    
    time[i] = endTime - startTime;
  }

  //printf("time (rdtsc) cycles:\n");
  unsigned long long  avg = 0.0;
  for(int i = 0; i < TIMES_MEASURED; ++i) {
    //printf("%d,%llu\n", i, time[i]);
    avg += time[i];
  }
  avg /= TIMES_MEASURED;
  
  printf("%u, %llu\n", bytes, avg);
  
  // cleanup
  cudaFree(dev_arr);
  cudaFreeHost(host_arr);
}

//  [HOST]  ---  ---  ---  sendRawBit  ---  ---  ---  ---  ---  [by meside]
void sendRawBit(int bit, float *dev_arr, float *host_arr, 
  const unsigned int bytes) 
{
  if (bit == 1) {
    for(int j = 0; j < TIMES_REPEAT; ++j) {
      // BEWARE you are not checking for errors! see cudaError_t checkCuda()
      cudaMemcpy(dev_arr,host_arr,bytes,cudaMemcpyHostToDevice);
    }
  } else {
    for(unsigned int i = 0; i < NOP_REPEAT * TIMES_REPEAT; ++i) {
      __asm__ __volatile__("nop");      
    }
    /*
    for(int i = 0; i < TIMES_REPEAT; ++i) {
      //usleep(NOP_DELAY);
      // struct timespec req = {0};
      // req.tv_sec = 0;
      // req.tv_nsec = 1L;
      // //nanosleep((const struct timespec[]){{0, 1L}}), (struct timespec *)NULL);
      // nanosleep(&req, (struct timespec *)NULL);
    }
    */
  }
}

//  0   1   1   0   1   0   1   0
// 0 1 1 0 1 0 0 1 1 0 0 1 1 0 0 1 
//  [HOST]  ---  ---  ---  encodeManchester  ---  ---  ---  ---  ---  [by meside]
void encodeManchester(int bit, float *dev_arr, float *host_arr, 
  const unsigned int bytes) 
{
  if(bit == 0) {
    for(unsigned int i = 0; i < NOP_REPEAT * TIMES_REPEAT; ++i) {
      __asm__ __volatile__("nop");      
    }
    // for(int i = 0; i < TIMES_REPEAT; ++i) {
    //   usleep(NOP_DELAY);
    // }
    for(int i = 0; i < TIMES_REPEAT; ++i) {
      // BEWARE you are not checking for errors! see cudaError_t checkCuda()
      cudaMemcpy(dev_arr,host_arr,bytes,cudaMemcpyHostToDevice);
    }
  }
  if(bit == 1) {
    for(int i = 0; i < TIMES_REPEAT; ++i) {
      // BEWARE you are not checking for errors! see cudaError_t checkCuda()
      cudaMemcpy(dev_arr,host_arr,bytes,cudaMemcpyHostToDevice);
    }
    for(unsigned int i = 0; i < NOP_REPEAT * TIMES_REPEAT; ++i) {
      __asm__ __volatile__("nop");      
    }
    // for(int i = 0; i < TIMES_REPEAT; ++i) {
    //   usleep(NOP_DELAY);
    // }
  }
  
}

//  [HOST]  ---  ---  ---  sendHeader  ---  ---  ---  ---  ---  [by meside]
void sendHeader(const unsigned int bytes) 
{
  // Pinned transfers Host to Device bandwidth (GB/s):
  //printf("bytes: %u\n", bytes);
  //printf(CLOCK_PER_SECOND);

  // host array
  float *host_arr;

  // device array
  float *dev_arr;

  // allocate and initialize
  checkCuda( cudaMallocHost((void**)&host_arr, bytes) );  // host pinned
  checkCuda( cudaMalloc((void**)&dev_arr, bytes) );       // device
  memset(host_arr, 0, bytes);

  // perform copies and report bandwidth
  // int header[8] = {0,1,1,0,1,0,0,1};
  int header[16] = {0,1,1,0,1,0,0,1,0,1,0,1,1,0,0,1};
  int headerSize = sizeof(header)/sizeof(header[0]);
  // printf("\t%d\n", headerSize);
  for(int i = 0; i < headerSize; ++i) {
    sendRawBit(header[i],dev_arr,host_arr,bytes);
    //encodeManchester(header[i],dev_arr,host_arr,bytes);
  }

  // cleanup
  // cudaFree(dev_arr);
  // cudaFreeHost(host_arr);
}

//  [HOST]  ---  ---  ---  sendMsg  ---  ---  ---  ---  ---  [by meside]
void sendMsg(const unsigned int bytes) 
{
  char *msgBuffer = NULL;
  size_t msgSize = 0;
  
  /* Open your_file in read-only mode */
  FILE *fp = fopen("msg/binMsg512.txt", "r");
  //FILE *fp = fopen("msg/binMsg1024.txt", "r");
  
  /* Get the buffer size */
  fseek(fp, 0, SEEK_END); /* Go to end of file */
  msgSize = ftell(fp); /* How many bytes did we pass ? */
  
  /* Set position of stream to the beginning */
  rewind(fp);
  
  /* Allocate the buffer (no need to initialize it with calloc) */
  msgBuffer = (char *) malloc((msgSize + 1) * sizeof(*msgBuffer)); 
  /* size + 1 byte for the \0 */
  
  /* Read the file into the buffer */
  fread(msgBuffer, msgSize, 1, fp); 
  /* Read 1 chunk of size bytes from fp into buffer */
  
  /* NULL-terminate the buffer */
  msgBuffer[msgSize] = '\0';
  
  /* Print it ! */
  // printf("%s\n", msgBuffer);

  // for(int i = 0; i < msgSize; ++i) {
  //   int tmp = (int) (msgBuffer[i] - '0');
  //   printf("%d",  tmp);
  // }

  // host array
  float *host_arr;

  // device array
  float *dev_arr;

  // allocate and initialize
  checkCuda( cudaMallocHost((void**)&host_arr, bytes) );  // host pinned
  checkCuda( cudaMalloc((void**)&dev_arr, bytes) );       // device
  memset(host_arr, 0, bytes);

  // perform copies and report bandwidth
  // int header[8] = {0,1,1,0,1,0,0,1};
  int header[16] = {0,1,1,0,1,0,0,1,0,1,0,1,1,0,0,1};
  int headerSize = sizeof(header)/sizeof(header[0]);
  // printf("\theaderSize: %d\n", headerSize);
  for(int i = 0; i < headerSize; ++i) {
    sendRawBit(header[i],dev_arr,host_arr,bytes);
  }
  for(int i = 0; i < msgSize; ++i) {
    int tmp = (int) (msgBuffer[i] - '0');
    sendRawBit(tmp,dev_arr,host_arr,bytes);
  }

  // cleanup
  cudaFree(dev_arr);
  cudaFreeHost(host_arr);

  // printf("\tMessage sent! Total size: %lu\n", 
  //         ((headerSize + msgSize)) );
}

//  [HOST]  ---  ---  ---  timespec diff ---  ---  ---  ---  ---  [by meside]
double diff(timespec start, timespec end)
{
  unsigned long temp1 = start.tv_sec * 1000000000 + start.tv_nsec;
  unsigned long temp2 = end.tv_sec * 1000000000 + end.tv_nsec;
  return (double) (temp2 - temp1); // in nanoseconds
}

//  [HOST]  ---  ---  ---  measureSendingOne  ---  ---  ---  ---  [by meside]
void measureSendingOne(const unsigned int bytes) 
{
  // host array
  float *host_arr;
  // device array
  float *dev_arr;

  // allocate and initialize
  checkCuda( cudaMallocHost((void**)&host_arr, bytes) );  // host pinned
  checkCuda( cudaMalloc((void**)&dev_arr, bytes) );       // device
  memset(host_arr, 0, bytes);

  // perform copies and report bandwidth
  int len = 100;
  unsigned int timers[len];
  timespec startTime, endTime;
  for(int i = 0; i < len; ++i) {
    clock_gettime(CLOCK_MONOTONIC, &startTime);
    sendRawBit(1,dev_arr,host_arr,bytes);
    clock_gettime(CLOCK_MONOTONIC, &endTime);
    timers[i] = diff(startTime, endTime);
  }

  unsigned int avgTimers = 0;
  for(int i = 0; i < len; ++i) {
    printf("%u\n", timers[i]);
    avgTimers += timers[i];
  }
  avgTimers /= len; 
  printf("mean %u\n", avgTimers);

  // cleanup
  // cudaFree(dev_arr);
  // cudaFreeHost(host_arr);
}

//  [HOST]  ---  ---  ---  sendPckt  ---  ---  ---  ---  ---  [by meside]
void sendPckt(const unsigned int bytes, int pcktSize) 
{
  char *msgBuffer = NULL;
  size_t msgSize = 0;
  
  /* Open your_file in read-only mode */
  FILE *fp = fopen("msg/binMsg.txt", "r");
  
  /* Get the buffer size */
  fseek(fp, 0, SEEK_END); /* Go to end of file */
  if(pcktSize <= 32) {
    msgSize = ftell(fp); /* How many bytes did we pass ? */
  } else {
    msgSize = (size_t) pcktSize;
  }
  
  /* Set position of stream to the beginning */
  rewind(fp);
  
  /* Allocate the buffer (no need to initialize it with calloc) */
  msgBuffer = (char *) malloc((msgSize + 1) * sizeof(*msgBuffer)); 
  /* size + 1 byte for the \0 */
  
  /* Read the file into the buffer */
  fread(msgBuffer, msgSize, 1, fp); 
  /* Read 1 chunk of size bytes from fp into buffer */
  
  /* NULL-terminate the buffer */
  msgBuffer[msgSize] = '\0';
  
  /* Print it ! */
  // printf("%s\n", msgBuffer);

  // for(int i = 0; i < msgSize; ++i) {
  //   int tmp = (int) (msgBuffer[i] - '0');
  //   printf("%d",  tmp);
  // }

  // host array
  float *host_arr;

  // device array
  float *dev_arr;

  // allocate and initialize
  checkCuda( cudaMallocHost((void**)&host_arr, bytes) );  // host pinned
  checkCuda( cudaMalloc((void**)&dev_arr, bytes) );       // device
  memset(host_arr, 0, bytes);

  // perform copies and report bandwidth
  // int header[8] = {0,1,1,0,1,0,0,1};
  int header[16] = {0,1,1,0,1,0,0,1,0,1,0,1,1,0,0,1}; // 0110100101011001
  int headerSize = sizeof(header)/sizeof(header[0]);
  int footer[16] = {1,0,0,1,0,1,1,0,1,0,1,0,0,1,1,0}; // 1001011010100110
  int footerSize = sizeof(footer)/sizeof(footer[0]);
  // printf("\theaderSize: %d\n", headerSize);
  for(int i = 0; i < headerSize; ++i) {
    sendRawBit(header[i],dev_arr,host_arr,bytes);
  }
  for(int i = 0; i < msgSize; ++i) {
    int tmp = (int) (msgBuffer[i] - '0');
    sendRawBit(tmp,dev_arr,host_arr,bytes);
  }
  for(int i = 0; i < footerSize; ++i) {
    sendRawBit(footer[i],dev_arr,host_arr,bytes);
  }

  // cleanup
  cudaFree(dev_arr);
  cudaFreeHost(host_arr);

  // printf("\tMessage sent! Total size: %lu\n", 
  //         ((headerSize + msgSize)) );
}

//  [HOST]  ---  ---  ---  ---  ---  ---  main  ---  ---  ---  ---  ---  ---
int main(int argc, char* argv[])
{
  const unsigned int bytes = 64 * 1024; //256 * 1024;
  //printf("\nCLOCKS_PER_SEC: %ld \n", CLOCKS_PER_SEC);

  int modeSelection = 0;
  int pcktSize = 0;

  if(argc > 1) {
    modeSelection = atoi(argv[1]);
    if(argc > 2) {
      pcktSize = atoi(argv[2]);
    }
  }

  if(modeSelection == 1) {
    printf("\n  bytes:               %u \n", bytes);
    printf("  TIMES_REPEAT:        %d \n", TIMES_REPEAT);
    printf("  infiniteCpy:\n");
    infiniteCpyPin(bytes);

  } else if(modeSelection == 2) {
    sendHeader(bytes);

  } else if(modeSelection == 3) {
    measureSendingOne(bytes);

  } else if(modeSelection == 5) {

    clock_t t = clock();  // start time
    sendMsg(bytes);
    t = clock() - t;      // end time 
    
    double timeTaken = ((double) t) / CLOCKS_PER_SEC;
    printf("timeTaken: %lf \n", timeTaken);

  } else if(modeSelection == 9) {
    unsigned int noOfBytes = 32 * 1024;
    printf("\n  bytes:               %u \n", noOfBytes);
    infiniteCpyPin(noOfBytes);

  } else if(modeSelection == 10) {
    unsigned int noOfBytes = 32 * 1024;
    printf("\n  bytes:               %u \n", noOfBytes);
    infiniteCpyPag(noOfBytes);

  } else if(modeSelection == 11) {
    // OBSERVE RELATION of TRANSFER SIZE to NO OF CYCLES
    unsigned int startSize = 4; //BYTE
    printf("  TIMES_MEASURED:        %d\n", TIMES_MEASURED);
    printf("\nBYTES, cycles\n");
    for(unsigned int i = startSize; i <= 1024*1024; i=i*2) {
      latencyPerSizePin(i);
      usleep(1000);
    }

  } else if(modeSelection == 25) {
    
    clock_t t = clock();  // start time
    sendPckt(bytes, pcktSize);
    t = clock() - t;      // end time 
    
    double timeTaken = ((double) t) / CLOCKS_PER_SEC;
    printf("timeTaken: %lf \n", timeTaken);

  } else {
    printf("\n  bytes:               %u\n", bytes);
    printf("  TIMES_REPEAT:        %d \n", TIMES_REPEAT);
    //printf("  NOP_DELAY:           %d \n", NOP_DELAY);
    printf("  NOP_REPEAT:          %d \n", NOP_REPEAT);

    printf("\nEnter one of the following arguments\n");
    printf(" 1: infiniteCpy()\n");
    printf(" 2: sendHeader()\n");
    printf(" 3: measureSendingOne()\n");
    printf(" 5: sendMsg()\n");
    printf(" 9: infiniteCpyPin(FIXED)\n");
    printf(" 10: infiniteCpyPag(FIXED)\n");
    printf(" 11: latencyPerSizePin()\n");
    printf(" 25: sendPckt()\n");

  }

  // sleep(2);
  // //printf("\narg: %d\n", v2);
  // printf("\nv1: %d\n", v1);
  // printf("\nv2: %d\n", v2);

  return 0;
}
