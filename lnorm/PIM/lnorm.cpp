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
  //Please attempt rmsnorm implementation before attempting this
 
  
  //TODO: Allocate source vector
  //TODO: Allocate destination vector (use pimAllocAssociated: libpimeval/src/libpimeval.h)
  //TODO: Allocate temporary vector (if needed) (use pimAllocAssociated)
 

  PimStatus status;

  //TODO: Copy source vector to PIM

  int32_t sum = 0;
  int32_t mean = 0;
  //TODO: Compute mean of the source vector (if there is any part to be run on CPU, time it)
  
  //TODO: Subtract mean from the source vector and store it a temporary vector
  
  //TODO: Compute square of the temporary vector and store it in the destination vector


  int32_t sum2 = 0;
  int32_t variance = 0;
  int32_t sqrt_var = 0;
  //TODO: Find variance of (X - mean)**2, time the CPU part
  

  //TODO: Scale the temporary vector with the square of the variance
  


  dst.resize(vectorLength);
  
  //TODO: Copy the destination vector to host

  //TODO: Free the PIM objects
 
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
