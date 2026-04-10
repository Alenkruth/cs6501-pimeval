// Test: C++ version of matrix vector multiplication
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include <iostream>
#include <vector>
#include <getopt.h>
#include <stdint.h>
#include <iomanip>
#if defined(_OPENMP)
#include <omp.h>
#endif

#include "util.h"
#include "libpimeval.h"
#include <chrono>
#include <cmath>

std::chrono::duration<double, std::milli> hostElapsedTime = std::chrono::duration<double, std::milli>::zero();

// Params ---------------------------------------------------------------------
typedef struct Params
{
  uint64_t vectorLength;
  char *configFile;
  char *inputFile;
  bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./rmsnorm.out [options]"
          "\n"
          "\n    -l    vectorLength (default=128 elements)"
          "\n    -c    dramsim config file"
          "\n    -i    input file containing two vectors (default=generates vector with random numbers)"
          "\n    -v    t = verifies PIM output with host output. (default=false)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.vectorLength = 128;
  p.configFile = nullptr;
  p.inputFile = nullptr;
  p.shouldVerify = false;

  int opt;
  while ((opt = getopt(argc, argv, "h:l:c:i:v:")) >= 0)
  {
    switch (opt)
    {
    case 'h':
      usage();
      exit(0);
      break;
    case 'l':
      p.vectorLength = strtoull(optarg, NULL, 0);
      break;
    case 'c':
      p.configFile = optarg;
      break;
    case 'i':
      p.inputFile = optarg;
      break;
    case 'v':
      p.shouldVerify = (*optarg == 't') ? true : false;
      break;
    default:
      fprintf(stderr, "\nUnrecognized option!\n");
      usage();
      exit(0);
    }
  }
  return p;
}

// Newton-Raphson iterative integer square root
uint32_t newton_sqrt(uint32_t x) {
  if (x == 0) return 0;  // Handle zero case

  uint32_t guess = x; // Initial guess
  uint32_t prev_guess = 0;

  while (guess != prev_guess) { // Continue until convergence
      prev_guess = guess;
      guess = (guess + x / guess) / 2; // Newton-Raphson iteration
  }

  return guess;
}

void rmsnorm(uint64_t vectorLength, std::vector<int> &srcVector, std::vector<int> &dst)
{
  //Allocate memory for the source vector
  PimObjId srcObj1 = pimAlloc(PIM_ALLOC_AUTO, vectorLength, PIM_INT32);
  if (srcObj1 == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  //Allocate memory for the destination vector (pimAllocAssociated) ensures it is allocated in the same region as srcObj1
  PimObjId dstObj = pimAllocAssociated(srcObj1, PIM_INT32);
  if (dstObj == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  PimStatus status;

  // Copy the source vector to the PIM device
  status = pimCopyHostToDevice((void *)srcVector.data(), srcObj1);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  // TODO: Square the element of the vector

  uint32_t sum = 0;
  // TODO: Sum of the squared elements
  
  auto start_cpu = std::chrono::high_resolution_clock::now();
  // TODO: Calculate scalar operations related mean and rms on CPU and time it

  auto stop_cpu = std::chrono::high_resolution_clock::now();
  hostElapsedTime += (stop_cpu - start_cpu);

  // TODO: Divide the source vector by the RMS value


  dst.resize(vectorLength);

  // Copy the result back to the host
  status = pimCopyDeviceToHost(dstObj, (void *)dst.data());
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
  }

  //Free PIM objects 
  pimFree(srcObj1);
  pimFree(dstObj);
}

int main(int argc, char *argv[])
{
  struct Params params = getInputParams(argc, argv);
  std::cout << "Running RMSNORM for vector of size: " << params.vectorLength << std::endl;

  std::vector<int> srcVector (params.vectorLength, 1), resultVector;

  if (params.shouldVerify) {
    if (params.inputFile == nullptr)
    {
      getVector(params.vectorLength, srcVector);
    }
    else
    {
      std::cout << "Reading from input file is not implemented yet." << std::endl;
      return 1;
    }
  }

  if (!createDevice(params.configFile))
  {
    return 1;
  }

  // PIM kernel (Complete all the TODOs in this kernel)
  rmsnorm(params.vectorLength, srcVector, resultVector);

  if (params.shouldVerify)
  {
    bool shouldBreak = false; // shared flag variable

    // verify result
      std::vector<int> result (params.vectorLength, 0);
      
      //rms norm
      uint32_t sum_sq = 0;

      // Compute sum of squares
      for (size_t i = 0; i < params.vectorLength; i++) {
          sum_sq += (uint32_t)(srcVector[i] * srcVector[i]); // Prevent overflow
      }

      // Compute mean squared value
      uint32_t mean_sq = sum_sq / params.vectorLength;

      // Compute RMS using Newton-Raphson square root
      uint32_t rms = newton_sqrt(mean_sq + 1); // +1 to prevent division by zero


      // Normalize each element: Y[i] = X[i] / RMS
      for (size_t i = 0; i < params.vectorLength; i++) {
          result[i] = srcVector[i] / (rms + 1);  // Prevent division by zero
      }

    for (size_t i = 0; i < params.vectorLength; i++)
    {
      if (result[i] != resultVector[i])
      {
        #pragma omp critical
        {
          if (!shouldBreak)
          { // check the flag again in a critical section
            std::cout << "Wrong answer: " << resultVector[i] << " (expected " << result[i] << ")" << std::endl;
            shouldBreak = true; // set the flag to true
          }
        }
      }
    }
    

    if (!shouldBreak) {
      std::cout << "\n\nCorrect Answer!!\n\n";
    }
  }

  pimShowStats();
  std::cout << "Host elapsed time: " << std::fixed << std::setprecision(3) << hostElapsedTime.count() << " ms." << endl;

  return 0;
}
