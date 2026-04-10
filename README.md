# cs6501-pimeval

PIMeval assignment: `lnorm` (Layer Normalization) and `rmsnorm` (RMS Normalization) workloads for PIMbench.

## Setup

Clone the repo and run the setup script:

```bash
git clone <repo-url>
cd assignment_pimeval
bash setup.sh -j8   # -j flag is optional, controls parallel build jobs
```

This will:
1. Initialize and clone the `PIMeval-PIMbench` submodule
2. Build PIMeval (`libpimeval` + all PIMbench workloads)
3. Symlink `lnorm/` and `rmsnorm/` into `PIMeval-PIMbench/PIMbench/` so the build system can find them

## Building the workloads

Build the PIM implementations:

```bash
cd PIMeval-PIMbench/PIMbench/lnorm/PIM
make perf USE_OPENMP=1
./lnorm.out -h   # show usage
```

```bash
cd PIMeval-PIMbench/PIMbench/rmsnorm/PIM
make perf USE_OPENMP=1
./rmsnorm.out -h
```

Build the CPU baselines:

```bash
cd lnorm/baselines/CPU && make
cd rmsnorm/baselines/CPU && make
```

## Project layout

```
.
├── setup.sh                  # one-time setup script
├── lnorm/                    # layer normalization workload (your code)
│   ├── PIM/
│   │   └── lnorm.cpp
│   └── baselines/CPU/
│       └── lnorm.cpp
├── rmsnorm/                  # RMS normalization workload (your code)
│   ├── PIM/
│   │   └── rmsnorm.cpp
│   └── baselines/CPU/
│       └── rmsnorm.cpp
└── PIMeval-PIMbench/         # submodule (do not edit)
    ├── libpimeval/
    └── PIMbench/
        ├── lnorm -> ../../lnorm   # symlink created by setup.sh
        └── rmsnorm -> ../../rmsnorm
```

Edit source files in `lnorm/` and `rmsnorm/` at the repo root. The symlinks let PIMbench's build system find them without copying files around.
