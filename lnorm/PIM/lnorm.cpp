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
//auto start_cpu, stop_cpu;

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
          "\nUsage:  ./lnorm.out [options]"
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

void lnorm(uint64_t vectorLength, std::vector<int> &srcVector, std::vector<int> &dst)
{
  // Allocate source vector
  PimObjId srcObj1 = pimAlloc(PIM_ALLOC_AUTO, vectorLength, PIM_INT32);
  if (srcObj1 == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  // Allocate destination vector
  PimObjId dstObj = pimAllocAssociated(srcObj1, PIM_INT32);
  if (dstObj == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  // Allocate temporary vector (stores x - mean)
  PimObjId tmpObj = pimAllocAssociated(srcObj1, PIM_INT32);
  if (tmpObj == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  PimStatus status;

  // Copy source vector to PIM
  status = pimCopyHostToDevice((void *)srcVector.data(), srcObj1);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  int32_t sum = 0;
  int32_t mean = 0;
  // Compute mean: reduce sum on PIM, divide on CPU
  status = pimRedSum(srcObj1, &sum);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }
  auto start_cpu = std::chrono::high_resolution_clock::now();
  mean = sum / (int32_t)vectorLength;
  auto stop_cpu = std::chrono::high_resolution_clock::now();
  hostElapsedTime += (stop_cpu - start_cpu);

  // Subtract mean from source vector: tmpObj[i] = srcObj1[i] - mean
  status = pimSubScalar(srcObj1, tmpObj, (uint64_t)mean);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  // Square the (x - mean) vector: dstObj[i] = tmpObj[i] * tmpObj[i]
  status = pimMul(tmpObj, tmpObj, dstObj);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  int32_t sum2 = 0;
  int32_t variance = 0;
  int32_t sqrt_var = 0;
  // Compute variance: reduce sum on PIM, divide and sqrt on CPU
  status = pimRedSum(dstObj, &sum2);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }
  auto start_cpu2 = std::chrono::high_resolution_clock::now();
  variance = sum2 / (int32_t)vectorLength;
  sqrt_var = newton_sqrt(variance + 1);
  if (sqrt_var == 0) sqrt_var = 1;
  auto stop_cpu2 = std::chrono::high_resolution_clock::now();
  hostElapsedTime += (stop_cpu2 - start_cpu2);

  // Divide (x - mean) by sqrt(variance): dstObj[i] = tmpObj[i] / sqrt_var
  status = pimDivScalar(tmpObj, dstObj, (uint64_t)sqrt_var);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  dst.resize(vectorLength);

  // Copy the destination vector to host
  status = pimCopyDeviceToHost(dstObj, (void *)dst.data());
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  // Free the PIM objects
  pimFree(srcObj1);
  pimFree(dstObj);
  pimFree(tmpObj);
 
}

int main(int argc, char *argv[])
{
  struct Params params = getInputParams(argc, argv);
  std::cout << "Running LNORM for vector of size: " << params.vectorLength << std::endl;

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

  // PIM kernel
  lnorm(params.vectorLength, srcVector, resultVector);

  if (params.shouldVerify)
  {
    bool shouldBreak = false; // shared flag variable

    // verify result

      std::vector<int> result (params.vectorLength, 0);
      std::vector<int> src_minus_mean (params.vectorLength, 0);
      std::vector<int> sq_src_minus_mean (params.vectorLength, 0);
      
      int32_t sum = 0;

      for (size_t i = 0; i < params.vectorLength; i++) {
          sum += srcVector[i];
      } 

      int32_t mean = sum / params.vectorLength; 

      for (size_t i = 0; i < params.vectorLength; i++) {
        src_minus_mean[i] = srcVector[i] - mean;  
      }

      for (size_t i = 0; i < params.vectorLength; i++) {
        sq_src_minus_mean[i] = (int32_t)(src_minus_mean[i]*src_minus_mean[i]);  
      }

      int32_t sum2 = 0;
      for (size_t i = 0; i < params.vectorLength; i++) {
        sum2 += sq_src_minus_mean[i];
      } 

      int32_t var = sum2/params.vectorLength;

      int32_t sqrt_var = newton_sqrt(var+1);
      if(sqrt_var==0){
        sqrt_var = 1;
      }

      // layer norm
      for (size_t i = 0; i < params.vectorLength; i++) {
          result[i] = src_minus_mean[i] / (sqrt_var);  // Prevent division by zero
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
