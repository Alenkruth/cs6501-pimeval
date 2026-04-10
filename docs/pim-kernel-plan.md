# Plan: Implement PIM Kernels for RMSNorm and LayerNorm

## Context

This is a PIM programming assignment (CS6501, due 2026-04-10). The task is to fill in TODO sections in two skeleton files that implement RMS Normalization and Layer Normalization using the PIMeval bank-level PIM simulator API. The skeletons already have CLI parsing, device creation, verification logic, and cleanup — we only need to fill in the PIM operations inside the `rmsnorm()` and `lnorm()` functions.

The CPU baselines and verification code in the skeletons define the exact integer arithmetic we must match.

---

## Equations

### RMSNorm

```
y_i = x_i / RMS(x)

where RMS(x) = sqrt(epsilon + (1/n) * sum(x_i^2))
```

With gamma=1 and epsilon=1 (matching the skeleton's `newton_sqrt(mean_sq + 1)`).

In integer arithmetic (matching verification code):
1. `sum_sq = sum(x_i * x_i)` for all i
2. `mean_sq = sum_sq / n`
3. `rms = newton_sqrt(mean_sq + 1)`
4. `y_i = x_i / (rms + 1)`

### LayerNorm

```
y = (x - E[x]) / sqrt(Var[x] + epsilon)

where E[x] = (1/n) * sum(x_i)
      Var[x] = (1/n) * sum((x_i - E[x])^2)
```

With gamma=1, beta=0, epsilon=1.

In integer arithmetic (matching verification code):
1. `sum = sum(x_i)` for all i
2. `mean = sum / n`
3. `src_minus_mean_i = x_i - mean` for all i
4. `sq_i = src_minus_mean_i * src_minus_mean_i` for all i
5. `sum2 = sum(sq_i)` for all i
6. `var = sum2 / n`
7. `sqrt_var = newton_sqrt(var + 1)`, clamped to min 1
8. `y_i = src_minus_mean_i / sqrt_var`

---

## PIM API Operations Needed

From the API description, the operations we'll use:

| Operation | API Call | Semantics |
|-----------|----------|-----------|
| Element-wise multiply | `pimMul(src1, src2, dest)` | `dest[i] = src1[i] * src2[i]` |
| Reduction sum | `pimRedSum(src, &sum)` | `sum = sum(src[i])` |
| Scalar subtract | `pimSubScalar(src, dest, scalar)` | `dest[i] = src[i] - scalar` |
| Scalar divide | `pimDivScalar(src, dest, scalar)` | `dest[i] = src[i] / scalar` |

Key constraint: `sqrt` and `division-to-get-mean` are scalar operations that cannot run on PIM — they must run on the CPU host. The skeleton already provides timing brackets (`start_cpu`/`stop_cpu`) for these scalar sections.

---

## Implementation: RMSNorm (`rmsnorm/PIM/rmsnorm.cpp`)

File: `rmsnorm/PIM/rmsnorm.cpp`

The skeleton already provides (lines 99-122):
- `srcObj1` allocated via `pimAlloc`
- `dstObj` allocated via `pimAllocAssociated`
- `srcVector` copied to `srcObj1` via `pimCopyHostToDevice`

### TODO 1 (line 124): Square the elements
```cpp
// TODO: Square the element of the vector
status = pimMul(srcObj1, srcObj1, dstObj);  // dstObj[i] = srcObj1[i] * srcObj1[i]
```
This computes x_i^2 for all elements in parallel on PIM, storing in `dstObj`.

### TODO 2 (line 127): Sum of squared elements
```cpp
// TODO: Sum of the squared elements
status = pimRedSum(dstObj, &sum);  // sum = sum(dstObj[i])
```
`pimRedSum` reduces the entire squared vector to a single scalar `sum` returned to host.

### TODO 3 (lines 130-131): Scalar CPU operations (mean, rms)
```cpp
// TODO: Calculate scalar operations related mean and rms on CPU and time it
uint32_t mean_sq = sum / vectorLength;
uint32_t rms = newton_sqrt(mean_sq + 1);
```
These are inherently scalar — can't be parallelized on PIM. The timer brackets already surround this section.

### TODO 4 (line 135): Divide source by RMS
```cpp
// TODO: Divide the source vector by the RMS value
status = pimDivScalar(srcObj1, dstObj, (uint64_t)(rms + 1));  // dstObj[i] = srcObj1[i] / (rms+1)
```
This divides every element of the original source vector by `(rms + 1)` in parallel on PIM. Result goes in `dstObj`, which is then copied to host at line 141.

### Data flow summary
```
srcObj1:  [x_0, x_1, ..., x_n]     (input, stays unchanged after square)
dstObj:   [x_0^2, x_1^2, ...]      (after pimMul)
       -> sum = pimRedSum(dstObj)    (scalar to host)
       -> CPU: mean_sq, rms          (scalar computation)
dstObj:   [x_0/(rms+1), ...]        (after pimDivScalar, final output)
```

---

## Implementation: LayerNorm (`lnorm/PIM/lnorm.cpp`)

File: `lnorm/PIM/lnorm.cpp`

The skeleton is more bare — we need to do the allocation and copies ourselves.

### TODO 1-3 (lines 102-104): Allocate PIM objects
```cpp
// Allocate source vector
PimObjId srcObj1 = pimAlloc(PIM_ALLOC_AUTO, vectorLength, PIM_INT32);
// Allocate destination vector
PimObjId dstObj = pimAllocAssociated(srcObj1, PIM_INT32);
// Allocate temporary vector
PimObjId tmpObj = pimAllocAssociated(srcObj1, PIM_INT32);
```
We need 3 objects: source (x), destination (output/scratch), temporary (stores x-mean intermediate). Error checking on each (abort if -1).

### TODO 4 (line 109): Copy source to PIM
```cpp
status = pimCopyHostToDevice((void *)srcVector.data(), srcObj1);
```

### TODO 5 (line 113): Compute mean
```cpp
status = pimRedSum(srcObj1, &sum);    // PIM: reduce to scalar
auto start_cpu = std::chrono::high_resolution_clock::now();
mean = sum / (int32_t)vectorLength;   // CPU: scalar division
auto stop_cpu = std::chrono::high_resolution_clock::now();
hostElapsedTime += (stop_cpu - start_cpu);
```

### TODO 6 (line 115): Subtract mean from source
```cpp
status = pimSubScalar(srcObj1, tmpObj, (uint64_t)mean);  // tmpObj[i] = srcObj1[i] - mean
```
Stores `(x - E[x])` in `tmpObj`. We keep this for the final division step.

### TODO 7 (line 117): Square the (x - mean) vector
```cpp
status = pimMul(tmpObj, tmpObj, dstObj);  // dstObj[i] = tmpObj[i]^2
```

### TODO 8 (line 123): Compute variance
```cpp
status = pimRedSum(dstObj, &sum2);   // PIM: reduce squared diffs
auto start_cpu2 = std::chrono::high_resolution_clock::now();
variance = sum2 / (int32_t)vectorLength;   // CPU: scalar division
sqrt_var = newton_sqrt(variance + 1);       // CPU: Newton-Raphson sqrt
if (sqrt_var == 0) sqrt_var = 1;            // prevent division by zero
auto stop_cpu2 = std::chrono::high_resolution_clock::now();
hostElapsedTime += (stop_cpu2 - start_cpu2);
```

### TODO 9 (line 126): Scale by inverse sqrt(variance)
```cpp
status = pimDivScalar(tmpObj, dstObj, (uint64_t)sqrt_var);  // dstObj[i] = tmpObj[i] / sqrt_var
```
Divides `(x - mean)` (in `tmpObj`) by `sqrt_var`, storing final result in `dstObj`.

### TODO 10 (line 132): Copy result to host
```cpp
status = pimCopyDeviceToHost(dstObj, (void *)dst.data());
```

### TODO 11 (line 134): Free PIM objects
```cpp
pimFree(srcObj1);
pimFree(dstObj);
pimFree(tmpObj);
```

### Data flow summary
```
srcObj1:  [x_0, x_1, ..., x_n]          (input, unchanged throughout)
       -> sum = pimRedSum(srcObj1)        (scalar to host)
       -> CPU: mean = sum / n
tmpObj:   [x_0-mean, x_1-mean, ...]      (after pimSubScalar, kept for final step)
dstObj:   [(x_0-mean)^2, ...]            (after pimMul)
       -> sum2 = pimRedSum(dstObj)        (scalar to host)
       -> CPU: variance, sqrt_var
dstObj:   [(x_0-mean)/sqrt_var, ...]     (after pimDivScalar, final output)
```

---

## Order of Implementation

1. **RMSNorm first** (simpler — only 4 TODOs, allocation already done)
2. **LayerNorm second** (more TODOs, follows same pattern but with extra subtract-mean step)

---

## Verification

After implementing each kernel:

```bash
cd PIMeval-PIMbench/PIMbench/rmsnorm/PIM
make clean && make perf USE_OPENMP=1
./rmsnorm.out -l 128 -v t
./rmsnorm.out -l 4096 -v t
./rmsnorm.out -l 8192 -v t
./rmsnorm.out -l 16384 -v t
```

```bash
cd PIMeval-PIMbench/PIMbench/lnorm/PIM
make clean && make perf USE_OPENMP=1
./lnorm.out -l 128 -v t
./lnorm.out -l 4096 -v t
./lnorm.out -l 8192 -v t
./lnorm.out -l 16384 -v t
```

Each should print `Correct Answer!!` when verification passes.

Also test with a config file to ensure PIM stats output works:
```bash
./rmsnorm.out -l 4096 -c ../../../configs/hbm/PIMeval_Bank_Rank8.cfg -v t
```

---

## Files Modified

- `rmsnorm/PIM/rmsnorm.cpp` — fill 4 TODOs (lines 124, 127, 130, 135)
- `lnorm/PIM/lnorm.cpp` — fill 7 TODOs (lines 102-104, 109, 113, 115, 117, 123, 126, 132, 134)
