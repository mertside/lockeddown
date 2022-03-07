// receiver.cu
// Creation 20210209
// by Mert SIDE

#include <stdio.h>
#include <assert.h>

#include <string.h>

#include <unistd.h> //usleep

#include <x86intrin.h>
#ifdef    NO_LFENCE
#define   lfence()
#else
#include <emmintrin.h>
#define   lfence()  _mm_lfence()
#endif

#define TIMES_MEASURED 5000
#define TIMES_REPEAT 1
#define THRESHOLD 30000
// #define THRESHOLD 100000
// 60000 for 128KB sender 
// 90000 for 256KB sender

#define TIMES_REPEAT_SENDER 6

#define SIG_CNT 200000 //200000 * 4 RXT6000  // for 32kb to last around 7sec
//#define SIG_CNT 50000 // for smaller packages
char sig[SIG_CNT];

//for 4 MB   | TIMES_REPEAT 10 | on RTX 2060 S
//  NO con: 12534316
//  CON:    36960689
//    tres: 24000000

//for 512 KB | TIMES_REPEAT 10 | on RTX 2060 S
//  NO con: 1758239
//  CON:    26225866 
//    tres: 12000000

//for 1 KB   | TIMES_REPEAT 10 | on RTX 2060 S
//  NO con: 206887
//  CON:    252949
//    tres: 230000

//for 4 B    | TIMES_REPEAT 10 | on RTX 2060 S
//  NO con: 204608
//  CON:    248188
//    tres: 225000

// Compiled assembly for Intel:
//                                objdump receiver.out -d | less
// Compiled assembly for Nvidia:
//                                cuobjdump -ptx receiver.out | less
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

//  [HOST]  ---  ---  ---  getPinnedH2DBandwidth  ---  ---  ---  [by meside]
void getPinnedH2DBandwidth(const unsigned int bytes) 
{
  printf("getPinnedH2DBandwidth...\n");
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
    
    //timeCollected[i] = diff(startTime, endTime);
    time[i] = endTime - startTime;
  }

  /*
  float avg = 0.0;
  for(int i = 0; i < TIMES_MEASURED; ++i) {
    printf("%f\n", bytes * 1e-6 / time[i]);
    avg += bytes * 1e-6 / time[i];
  }
  avg /= TIMES_MEASURED;
  printf("mean (GB/s) = %f\n", avg);
  */

  //printf("time (rdtsc) cycles:\n");
  unsigned long long  avg = 0.0;
  for(int i = 0; i < TIMES_MEASURED; ++i) {
    printf("%d,%llu\n", i, time[i]);
    avg += time[i];
  }
  avg /= TIMES_MEASURED;
  
  printf("mean (cycles) = %llu\n", avg);
  
  // cleanup
  cudaFree(dev_arr);
  cudaFreeHost(host_arr);
  printf("getPinnedH2DBandwidth.\n");
}

//  [HOST]  ---  ---  ---  getPageableH2DBandwidth  ---  ---  ---  [by meside]
void getPageableH2DBandwidth(const unsigned int bytes) 
{
  printf("getPageableH2DBandwidth...\n");
  // host array
  float *host_arr;
  // device array
  float *dev_arr;

  // allocate and initialize
  host_arr = (float*)malloc(bytes);                       // host pageable
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
      checkCuda( cudaMemcpy(dev_arr, host_arr, bytes, cudaMemcpyHostToDevice) );
    }

    // Ensure everything has been executed thus far
    lfence();
    // Stop timing.
    endTime = __rdtsc();
    // Ensure we have the counter value before proceeding 
    lfence();
    
    //timeCollected[i] = diff(startTime, endTime);
    time[i] = endTime - startTime;
  }

  /*
  float avg = 0.0;
  for(int i = 0; i < TIMES_MEASURED; ++i) {
    printf("%f\n", bytes * 1e-6 / time[i]);
    avg += bytes * 1e-6 / time[i];
  }
  avg /= TIMES_MEASURED;
  printf("mean (GB/s) = %f\n", avg);
  */

  //printf("time (rdtsc) cycles:\n");
  unsigned long long  avg = 0.0;
  for(int i = 0; i < TIMES_MEASURED; ++i) {
    printf("%d,%llu\n", i, time[i]);
    avg += time[i];
  }
  avg /= TIMES_MEASURED;
  
  printf("mean (cycles) = %llu\n", avg);

  // cleanup
  cudaFree(dev_arr);
  free(host_arr);
}

//  [HOST]  ---  ---  ---  getAvgCycForPin  ---  ---  ---  [by meside]
void getAvgCycForPin(const unsigned int bytes) 
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

//  [HOST]  ---  ---  ---  compareThreshold  ---  ---  ---  [by meside]
int compareThreshold(float *dev_arr, float *host_arr, const unsigned int bytes) 
{
  unsigned long long timer = 0;
  unsigned long long startTime, endTime;
  
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
  
  timer = endTime - startTime;

  if(timer > THRESHOLD)
    return 1;
  else
    return 0;
}

//  [HOST]  ---  ---  ---  decodeManchester  ---  ---  ---  [by meside]
void decodeManchester(const unsigned int  bytes) 
{
  // NOT TESTED !!!
  printf("\ndecodeManchester...\n");

  // host array
  float *host_arr;
  // device array
  float *dev_arr;
  // allocate and initialize
  checkCuda( cudaMallocHost((void**)&host_arr, bytes) );  // host pinned
  checkCuda( cudaMalloc((void**)&dev_arr, bytes) );       // device
  memset(host_arr, 0, bytes);

  //int i = 0;
  int priorPhase = 0;
  int currentPhase = 0;
  //int state = 0;

  int bufferSize = 24;
  int buffer[bufferSize]; 
  memset(buffer, 0, (bufferSize * sizeof(int)));

  //  0   1   1   0   1   0   1   0
  // 0 1 1 0 1 0 0 1 1 0 0 1 1 0 0 1 
  //int oneCounter = 0;
  while(1) {
    priorPhase = compareThreshold(dev_arr, host_arr, bytes);
    currentPhase = compareThreshold(dev_arr, host_arr, bytes);

    //buffer[i] = received;
    printf("%d ", currentPhase);
    if(priorPhase < currentPhase) {
      //state = 0;
      printf("*0 ");
    } else if (priorPhase > currentPhase) {
      //state = 0;
      printf("*1 ");
    } 
    /*
    if(state == 0 && priorPhase < currentPhase) {
      //state = 0;
      //printf("*0 ");
    } else if (state == 0 && priorPhase > currentPhase) {
      // state = 0;
      // printf("*0 ");
    } else if (state == 1 && priorPhase < currentPhase) {
      // state = 0;
      //printf("*1 ");
    } else if (state == 1 && priorPhase > currentPhase) {
      // state = 1;
      // printf("*1 ");
    }
    */

    //priorPhase = currentPhase;
    //i = (i + 1) % bufferSize;
  }

  // cleanup
  cudaFree(dev_arr);
  cudaFreeHost(host_arr);
}

//  [HOST]  ---  ---  ---  receiveSignal  ---  ---  ---  [by meside]
void receiveSignal(const unsigned int  bytes) 
{
  //printf("bytes: %u\n", bytes);
  // host array
  float *host_arr;
  // device array
  float *dev_arr;
  // allocate and initialize
  checkCuda( cudaMallocHost((void**)&host_arr, bytes) );  // host pinned
  checkCuda( cudaMalloc((void**)&dev_arr, bytes) );       // device
  memset(host_arr, 0, bytes);

  int received = 2;

  for (int i = 0; i < SIG_CNT; ++i) {
    received = compareThreshold(dev_arr, host_arr, bytes);
    if(received == 1) {
      sig[i] = '1';
    } else {
      sig[i] = '0';      
    }
  }

  // cleanup
  cudaFree(dev_arr);
  cudaFreeHost(host_arr);
}

//  [HOST]  ---  ---  ---  receiveInfPin  ---  ---  ---  [by meside]
void receiveInfPin(const unsigned int  bytes) 
{
  printf("\nreceiveInfPin...\n");

  // host array
  float *host_arr;
  // device array
  float *dev_arr;
  // allocate and initialize
  checkCuda( cudaMallocHost((void**)&host_arr, bytes) );  // host pinned
  checkCuda( cudaMalloc((void**)&dev_arr, bytes) );       // device
  memset(host_arr, 0, bytes);

  while(1) {
    compareThreshold(dev_arr, host_arr, bytes);
  }

  // cleanup
  cudaFree(dev_arr);
  cudaFreeHost(host_arr);

  printf("\nreceiveInfPin.\n");
}

//  [HOST]  ---  ---  ---  receiveInfPag  ---  ---  ---  [by meside]
void receiveInfPag(const unsigned int  bytes) 
{
  printf("\nreceiveInfPag...\n");

  // host array
  float *host_arr;
  // device array
  float *dev_arr;
  // allocate and initialize
  host_arr = (float*)malloc(bytes);                       // host pageable
  // checkCuda( cudaMallocHost((void**)&host_arr, bytes) );  // host pinned
  checkCuda( cudaMalloc((void**)&dev_arr, bytes) );       // device
  memset(host_arr, 0, bytes);

  while(1) {
    compareThreshold(dev_arr, host_arr, bytes);
  }

  // cleanup
  free(dev_arr);
  cudaFreeHost(host_arr);

  printf("\nreceiveInfPag.\n");
}

//  [HOST]  ---  ---  ---  printSignal  ---  ---  ---  [by meside]
void printSignal(const unsigned int  bytes) 
{
  receiveSignal(bytes);
  printf("\nprintSignal...\n");
  for (int i = 0; i < SIG_CNT; ++i) {
    printf("%c", sig[i]);
  }
  printf("\nprintSignal.\n");
}
//  [HOST]  ---  ---  ---  receivePacket  ---  ---  ---  [by meside]
void receivePacket(char buf[], long bufSize) 
{
  // printf("\nreceivePacket...\n");
  // to store packet
  // long packetSize = 8 * 28; // in bits including header
  //long packetSize = 41000; // 40800; // for complete message
  long packetSize = 2048; // for testing smaller packages 
  char packet[packetSize]; 
  memset(packet, '*', (packetSize * sizeof(char)));
  
  //char header[] = "01101001";
  char header[] = "0110100101011001";

  char* locationPtr = strstr(buf, header);

  if(locationPtr == NULL) {
    printf("\n! Header NOT found !\n");
  } else {
    long locationFound = ((long)locationPtr-(long)buf);
    // printf("\nHeader '%s' found at %ld.\n", header, locationFound);  
    
    int j = 0;
    long endOfPacket = locationFound + packetSize;
    if (endOfPacket < bufSize) {
      for(long i = locationFound; i < endOfPacket; ++i) {
        packet[j] = buf[i];
        printf("%c", packet[j]);
        j++;
      }
      printf("\n");
      // printf("\nPacket Received SUCCESSFULLY.\n");
    } else {
      printf("\n! Packet Incomplete !\n");
    }
  }
  // printf("\nreceivePacket.\n");
}

//  [HOST]  ---  ---  ---  interpreter  ---  ---  ---  [by meside]
void interpreter(const unsigned int  bytes) 
{
  printf("\ninterpreter...\n");

  receiveSignal(bytes);

  int hiCount = 0;
  int loCount = 0;

  int senderRepetition = TIMES_REPEAT_SENDER; //*2;
  int minCount = senderRepetition - 1;
  int maxCount = senderRepetition + 1;

  // to store interpreted data
  long bufferSize = SIG_CNT / minCount;
  char buffer[bufferSize]; 
  memset(buffer, '*', (bufferSize * sizeof(char)));

  int j = 0;
  for (int i = 0; i < SIG_CNT; ++i) {
    //printf("%d", sig[i]);
    
    // ignore lenghty continuous signal
    if(loCount > maxCount) {
      loCount = 0;
    }
    if(hiCount > maxCount) {
      hiCount = 0;
    }

    // handle high
    if(sig[i] == '1') {
      loCount=0; 
      hiCount++;
    } 
    if(hiCount >= minCount) {
      buffer[j] = '1';
      //j = (j+1)%bufferSize;
      hiCount = 0;
      j++;
    }
    // handle low
    if(sig[i] == '0') {
      hiCount = 0;
      loCount++;
    } 
    if(loCount >= minCount) {
      buffer[j] = '0';
      //j = (j+1)%bufferSize;
      loCount = 0;
      j++;
    }
  }

  bufferSize = j;
  //for (int i = 0; i < bufferSize; ++i) {
  //  printf("%c", buffer[i]);
  //}
  //printf("%s", buffer);
  printf("\ninterpreter.\n");

  receivePacket(buffer, bufferSize);
}

//  [HOST]  ---  ---  ---  receivePacketFromSignal  ---  ---  ---  [by meside]
void receivePacketFromSignal(const unsigned int  bytes) 
{
  // printf("\nreceivePacketFromSignal...\n");
  receiveSignal(bytes);
  //printSignal(bytes);
  receivePacket(sig, SIG_CNT);
  // printf("\nreceivePacketFromSignal.\n");
}

//  [HOST]  ---  ---  ---  ---  fingerprinting  ---  ---  ---  --- [by meside]
void fingerprinting(const unsigned int bytes, const unsigned int samples, 
  const unsigned int repeat) 
{
  // printf("fingerprinting...\n");

  // host array
  float *host_arr;
  // device array
  float *dev_arr;

  // allocate and initialize
  checkCuda( cudaMallocHost((void**)&host_arr, bytes) );  // host pinned
  checkCuda( cudaMalloc((void**)&dev_arr, bytes) );       // device
  memset(host_arr, 0, bytes);

  unsigned long long time[samples];
  unsigned long long startTime, endTime;
  // perform copies and report bandwidth
  for(unsigned int i = 0; i < samples; ++i) {
    // First, let's ensure our CPU executes everything thus far
    lfence();
    // Start timing
    startTime = __rdtsc();
    // Ensure timing starts before we call the function
    lfence();
    
    for(unsigned int j = 0; j < repeat; ++j) {
      // BEWARE you are not checking for errors! see cudaError_t checkCuda()
      cudaMemcpy(dev_arr, host_arr, bytes, cudaMemcpyHostToDevice);
    }

    // Ensure everything has been executed thus far
    lfence();
    // Stop timing.
    endTime = __rdtsc();
    // Ensure we have the counter value before proceeding 
    lfence();
    
    //timeCollected[i] = diff(startTime, endTime);
    time[i] = endTime - startTime;
  }

  //printf("time (rdtsc) cycles:\n");
  unsigned long long  avg = 0.0;
  for(unsigned int i = 0; i < samples; ++i) {
    printf("%u,%llu\n", i, time[i]);
    avg += time[i];
  }
  avg /= samples;
  
  // printf("mean (rdtsc) = %llu\n", avg);
  
  // cleanup
  cudaFree(dev_arr);
  cudaFreeHost(host_arr);
}
//  [HOST]  ---  ---  ---  receivePacket_2  ---  ---  ---  [by meside]
void receivePacket_2(char buf[], long bufSize) 
{
  // printf("\nreceivePacket...\n");
  // to store packet
  // long packetSize = 8 * 28; // in bits including header
  
  // long packetSize = 41000; //40800;
  // char packet[packetSize]; 
  // memset(packet, '*', (packetSize * sizeof(char)));
  
  //char header[] = "01101001";
  char header[] = "0110100101011001";
  char footer[] = "1001011010100110";

  char* headerLocationPtr = strstr(buf, header);
  char* footerLocationPtr = strstr(buf, footer);

  if(headerLocationPtr == NULL) {
    printf("\n! Header NOT found !\n");
  } else if(footerLocationPtr == NULL && 
            ((long)footerLocationPtr <= (long)headerLocationPtr) ) {
    printf("\n! Footer NOT found !\n");
  } else {
    long headerLocationFound = ((long)headerLocationPtr-(long)buf);
    long footerLocationFound = ((long)footerLocationPtr-(long)buf);
    // printf("\nHeader '%s' found at %ld.\n", header, headerLocationFound);  
    
    long packetSize = footerLocationFound - headerLocationFound + 64; 
    char packet[packetSize]; 
    memset(packet, '*', (packetSize * sizeof(char)));

    int j = 0;
    // long endOfPacket = headerLocationFound + packetSize;
    long endOfPacket = footerLocationFound;

    if (endOfPacket < bufSize) {
      for(long i = headerLocationFound; i < endOfPacket; ++i) {
        packet[j] = buf[i];
        printf("%c", packet[j]);
        j++;
      }
      printf("\n");
      // printf("\nPacket Received SUCCESSFULLY.\n");
    } else {
      printf("\n! Packet Incomplete !\n");
    }
  }
  // printf("\nreceivePacket.\n");
}

//  [HOST]  ---  ---  ---  receivePacketFromSignal_2  ---  ---  ---  [by meside]
void receivePacketFromSignal_2(const unsigned int  bytes) 
{
  // printf("\nreceivePacketFromSignal_2...\n");
  receiveSignal(bytes);
  //printSignal(bytes);
  receivePacket_2(sig, SIG_CNT);
  // printf("\nreceivePacketFromSignal_2.\n");
}

//  [HOST]  ---  ---  ---  ---  ---  ---  main  ---  ---  ---  ---  ---  ---
int main(int argc, char* argv[])
{
  // printf("main...\n");
  const unsigned int bytes = 32 * 1024; // 32 KB
  int modeSelection = 0;

  if(argc > 1) {
    modeSelection = atoi(argv[1]);
  }
  
  if (modeSelection != 5 && modeSelection != 7) {
    printf("\n  bytes:                 %u \n", bytes);
    printf("  TIMES_REPEAT:          %d \n", TIMES_REPEAT);
    printf("  THRESHOLD:             %d \n", THRESHOLD);
  }

  if(modeSelection == 1) {
    // DETERMINE THRESHOLD
    printf("  TIMES_MEASURED:        %d\n", TIMES_MEASURED);
    getPinnedH2DBandwidth(bytes);

  } else if(modeSelection == 2) {
    // DEBUGING: receive signal
    printSignal(bytes);

  } else if(modeSelection == 3) {
    // run while sender measures SPEED OF SENDING 1
    receiveInfPin(bytes);

  } else if(modeSelection == 4) {
    // COVERT CHANNEL RECEIVER if sender repeats a signal for robustness
    printf("  TIMES_REPEAT_SENDER:   %d \n", TIMES_REPEAT_SENDER);
    interpreter(bytes);

  } else if(modeSelection == 5) {
    // COVERT CHANNEL RECEIVER
    receivePacketFromSignal(bytes);

  } else if(modeSelection == 6) {
    // DECODE MANCESTER CODE  ---  ---  BROKEN  ---  ---  NOT TESTED !!!
    decodeManchester(bytes);

  } else if(modeSelection == 7) {
    // WEBSITE FINGERPRINTING  ---  ---  ---  CONFIG.: G 
    unsigned int noOfBytes = 32 * 1024;
    unsigned int samples = 5000;
    unsigned int rep = 200; //(RTX2080) // 150;(GTX1080 & RTX2060S)

    clock_t t = clock();  // start time
    fingerprinting(noOfBytes, samples, rep);
    t = clock() - t;      // end time 
    
    double timeTaken = ((double) t) / CLOCKS_PER_SEC;
    // printf("timeTaken: %lf \n", timeTaken);

  } else if(modeSelection == 9) {
    // OBSERVE RELATION of TRANSFER SIZE to NO OF CYCLES
    unsigned int noOfBytes = 512;
    printf("  TIMES_MEASURED:        %d\n", TIMES_MEASURED);
    printf("\nBYTES, cycles\n");
    for(unsigned int i = noOfBytes; i <= 16*1024*1024; i=i*2) {
      getAvgCycForPin(i);
      usleep(1000);
    }

  } else if(modeSelection == 11) {
    // Pageable Bandwidth
    printf("  TIMES_MEASURED:        %d\n", TIMES_MEASURED);
    getPageableH2DBandwidth(bytes);

  } else if(modeSelection == 20) {
    // OBSERVE RELATION of TRANSFER SIZE to NO OF CYCLES
    unsigned int startSize = 4; //BYTE
    printf("  TIMES_MEASURED:        %d\n", TIMES_MEASURED);
    
    printf("\ngetPinnedH2DBandwidth\n");
    for(unsigned int i = startSize; i <= 1*1024*1024; i=i*2) {
      printf("\nbytes: %u\n", i);
      getPinnedH2DBandwidth(i);
      usleep(1000);
    }

    printf("\ngetPageableH2DBandwidth\n");
    for(unsigned int i = startSize; i <= 1*1024*1024; i=i*2) {
     printf("\nbytes: %u\n", i);
      getPageableH2DBandwidth(i);
      usleep(1000);
    }

  } else if(modeSelection == 23) {
    // run while sender measures SPEED OF SENDING 1
    receiveInfPag(bytes);

  } else if(modeSelection == 25) {
    // COVERT CHANNEL RECEIVER
    receivePacketFromSignal_2(bytes);

  } else {
    // MENU
    printf("  TIMES_MEASURED:        %d\n", TIMES_MEASURED);
    printf("  TIMES_REPEAT_SENDER:   %d \n", TIMES_REPEAT_SENDER);

    printf("\nEnter one of the following arguments\n");
    printf(" 1: getPinnedH2DBandwidth()\n");
    printf(" 2: printSignal()\n");
    printf(" 3: receiveInfPin()\n");
    printf(" 4: interpreter()\n");
    printf(" 5: receivePacketFromSignal()\n");
    printf(" 6: decodeManchester()\n");
    printf(" 7: fingerprinting()\n");
    printf(" 9: getAvgCycForPin()\n");
    printf(" 11: getPageableH2DBandwidth()\n");
    printf(" 20: getPinnedH2DBandwidth(4)\n     getPageableH2DBandwidth(4)\n");
    printf(" 23: receiveInfPag()\n");
    printf(" 25: receivePacketFromSignal_2()\n");

  }
  
  return 0;
}
