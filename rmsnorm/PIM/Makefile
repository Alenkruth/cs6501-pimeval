# Makefile: C++ version of matrix vector multiplication
# Copyright (c) 2024 University of Virginia
# This file is licensed under the MIT License.
# See the LICENSE file in the root of this repository for more details.

# Find PROJ_ROOT: works both when this dir lives inside PIMbench (or is
# symlinked there) and when built from the repo root directly.
ifneq ($(wildcard ../../../Makefile.common),)
  PROJ_ROOT = ../../..
else ifneq ($(wildcard ../../PIMeval-PIMbench/Makefile.common),)
  PROJ_ROOT = ../../PIMeval-PIMbench
else
  $(error Cannot find Makefile.common — run setup.sh or set PROJ_ROOT manually)
endif
include ${PROJ_ROOT}/Makefile.common

# make USE_OPENMP=1
USE_OPENMP ?= 0
ifeq ($(USE_OPENMP),1)
	CXXFLAGS += -fopenmp
endif

CXXFLAGS += -fsanitize=address -g -fno-omit-frame-pointer

EXEC := rmsnorm.out
SRC := rmsnorm.cpp

debug perf dramsim3_integ: $(EXEC)

$(EXEC): $(SRC) $(DEPS)
	$(CXX) $< $(CXXFLAGS) -o $@

clean:
	rm -rf $(EXEC) *.dSYM
