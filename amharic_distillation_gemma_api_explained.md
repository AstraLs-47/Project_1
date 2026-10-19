# Comprehensive Line-by-Line Guide: Custom Weight-Only Quantizer (Qwen3-0.6B)

This guide provides a complete, beginner-friendly, line-by-line breakdown of every single code cell in the [`qwen3_custom_quantizer_v2.ipynb`](file:///c:/Users/User/Downloads/Second%20draft%20quantization/qwen3_custom_quantizer_v2.ipynb) notebook. It is designed to prepare you to ace your evaluation and deliver a clear, confident presentation.

---

## Table of Contents
1. [Core Concepts: What is Quantization and Why Do It?](#1-core-concepts-what-is-quantization-and-why-do-it)
2. [The Core Quantization Mathematics](#2-the-core-quantization-mathematics)
3. [Line-by-Line Code Breakdown for Every Code Cell](#3-line-by-line-code-breakdown-for-every-code-cell)
   - [Cell 2: Runtime Environment Diagnostics](#cell-2-runtime-environment-diagnostics)
   - [Cell 4: Loading the Baseline Model & Prompt Test](#cell-4-loading-the-baseline-model--prompt-test)
   - [Cell 6: Core Quantization Math Functions (`quantizer_core.py` Part 1)](#cell-6-core-quantization-math-functions-quantizer_corepy-part-1)
   - [Cell 7: Mathematical Unit Testing & Formula Verification](#cell-7-mathematical-unit-testing--formula-verification)
   - [Cell 9: INT4 Bit-Packing Engine (`quantizer_core.py` Part 2)](#cell-9-int4-bit-packing-engine-quantizer_corepy-part-2)
   - [Cell 10: Bit-Packing Verification Tests](#cell-10-bit-packing-verification-tests)
   - [Cell 12: `QuantizedLinear` Layer & Model Transformation (`quantizer_core.py` Part 3)](#cell-12-quantizedlinear-layer--model-transformation-quantizer_corepy-part-3)
   - [Cell 13: Memory Measurement & Standalone SafeTensors Checkpointing (`quantizer_core.py` Part 4)](#cell-13-memory-measurement--standalone-safetensors-checkpointing-quantizer_corepy-part-4)
   - [Cell 14: Inspecting Real Weights Across Configurations A through F](#cell-14-inspecting-real-weights-across-configurations-a-through-f)
   - [Cell 16: Evaluation Benchmarking Suite (Perplexity, Speed, Task Sanity)](#cell-16-evaluation-benchmarking-suite-perplexity-speed-task-sanity)
   - [Cell 18: Running the Benchmark Sweep](#cell-18-running-the-benchmark-sweep)
   - [Cell 20: Results Analysis, Comparison Tables & CSV Export](#cell-20-results-analysis-comparison-tables--csv-export)
   - [Cell 22: Selecting the Winning Configuration](#cell-22-selecting-the-winning-configuration)
   - [Cell 24: Checkpoint Export, Reload & Bit-Exact Verification](#cell-24-checkpoint-export-reload--bit-exact-verification)
4. [Evaluator Q&A and Presentation Defense](#4-evaluator-qa-and-presentation-defense)

---

# 1. Core Concepts: What is Quantization and Why Do It?

### 1.1 The Memory Problem in Large Language Models (LLMs)
Neural networks are mathematical functions composed of millions or billions of parameters ("weights"). By default:
- **FP32 (32-bit Floating Point)**: Each weight uses **4 bytes** (32 bits). A 0.6 billion parameter model like Qwen3-0.6B takes:
  $$\approx 600,000,000 \times 4 \text{ bytes} \approx 2.4 \text{ GB of storage}.$$
- **FP16 / BF16 (16-bit Floating Point)**: Each weight uses **2 bytes** (16 bits) $\approx 1.2 \text{ GB}$.
- **INT8 (8-bit Integer)**: Each weight uses **1 byte** (8 bits) $\approx 600 \text{ MB}$.
- **INT4 (4-bit Integer)**: Each weight uses **0.5 bytes** (4 bits) $\approx 300 \text{ MB}$.

When running an LLM on a consumer device (like a laptop with 16 GB of RAM where only ~8 GB is usable), we cannot afford huge memory footprints.

### 1.2 What is Weight-Only Quantization (W8A16 / W4A16)?
- **Weights (W)**: Stored in memory as compact integers (**INT8** or **INT4**).
- **Activations (A)**: Kept in full 16-bit floating point (**FP16** or **BF16**).
- **How it runs**: During text generation, the GPU loads the compact integer weights from RAM/VRAM, converts them on-the-fly back into floating-point numbers, and performs standard matrix multiplication ($Y = X \cdot W^T$).
- **Why this works**: In LLM inference with batch size 1 (a single user chatting), the bottleneck is **Memory Bandwidth** (the speed of transferring weights from memory into the processor), not raw compute. Reducing weight storage by 60%–75% cuts memory traffic dramatically and allows large models to fit in consumer memory.

---

# 2. The Core Quantization Mathematics

Quantization maps a continuous real floating-point number range $[w_{min}, w_{max}]$ to a discrete integer grid $[q_{min}, q_{max}]$.

### 2.1 Integer Ranges
For a signed $b$-bit integer:
$$q_{min} = -2^{b-1}, \quad q_{max} = 2^{b-1} - 1$$
- For **INT8** ($b=8$): $q_{min} = -128, \quad q_{max} = 127$
- For **INT4** ($b=4$): $q_{min} = -8, \quad q_{max} = 7$

### 2.2 Symmetric vs. Asymmetric Quantization

#### Symmetric Quantization
- **Concept**: We force the zero point $Z = 0$. The integer grid is centered symmetrically around zero.
- **Scale ($S$)**: We look at the largest absolute value in the weights:
  $$S = \frac{\max(|w_{min}|, |w_{max}|)}{q_{max}}$$
- **Quantization formula**:
  $$q = \text{clamp}\left(\text{round}\left(\frac{x}{S}\right), q_{min}, q_{max}\right)$$
- **Dequantization formula**:
  $$\hat{x} = S \cdot q$$
- **Pros & Cons**: Very fast and requires no storage for zero points. But if the weight distribution is skewed (e.g. mostly positive numbers), symmetric quantization wastes representation space.

#### Asymmetric Quantization
- **Concept**: Weights can have an offset zero point $Z$. Real-world $0.0$ maps to an exact integer on the grid.
- **Scale ($S$)**:
  $$S = \frac{w_{max} - w_{min}}{q_{max} - q_{min}}$$
- **Zero Point ($Z$)**:
  $$Z = \text{clamp}\left(\text{round}\left(q_{min} - \frac{w_{min}}{S}\right), q_{min}, q_{max}\right)$$
- **Quantization formula**:
  $$q = \text{clamp}\left(\text{round}\left(\frac{x}{S} + Z\right), q_{min}, q_{max}\right)$$
- **Dequantization formula**:
  $$\hat{x} = S \cdot (q - Z)$$
- **Why clamp $w_{min} \le 0$ and $w_{max} \ge 0$?** We must ensure real $0.0$ is inside $[w_{min}, w_{max}]$ so that exact zero padding has zero quantization error.

### 2.3 Granularity: Per-Tensor vs. Per-Channel vs. Per-Group
- **Per-Tensor**: Only 1 scale factor for an entire matrix (e.g., $1024 \times 1024$). If just one weight is large, it inflates the scale and destroys precision for all other 1,000,000 weights.
- **Per-Channel (Per-Row)**: 1 scale factor for each output row. This isolates weight outliers to their specific neurons.
- **Per-Group (e.g. Group-128)**: 1 scale factor for every consecutive 128 weights in a row. This provides local resolution against outlier spikes and is the standard for 4-bit quantization.

---

# 3. Line-by-Line Code Breakdown for Every Code Cell

---

### Cell 2: Runtime Environment Diagnostics

#### What this cell does:
Installs required packages, imports necessary libraries, and prints hardware specifications (Python, PyTorch, RAM, Disk, and GPU/BF16 support).

```python
!pip install -q "transformers>=4.51.0" accelerate safetensors psutil pandas tabulate datasets
```
- `!pip install -q ...`: Runs a shell command inside the notebook environment.
- `-q`: Quiet mode (avoids cluttering the notebook with hundreds of install lines).
- `"transformers>=4.51.0"`: Hugging Face transformers library containing the Qwen3 model architecture.
- `accelerate`: Hugging Face library used for hardware-accelerated model loading (`device_map="cuda"`).
- `safetensors`: A fast, secure binary storage format for saving and loading tensors without using Python pickle.
- `psutil`: Process and system utilities library used to query available CPU RAM and disk capacity.
- `pandas` and `tabulate`: Used to format diagnostic metrics and benchmark results into clean Markdown tables.
- `datasets`: Hugging Face library used to download the standard WikiText-2 language evaluation benchmark.

```python
import sys
import os
import time
import gc
import math
import shutil
import psutil
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
```
- `import sys, os, time, gc, math, shutil`: Standard Python utility modules for system version (`sys`), file directories (`os`), benchmark timers (`time`), garbage collection (`gc`), exponential math (`math`), and disk management (`shutil`).
- `import torch`: Imports PyTorch, the core deep learning library.
- `import torch.nn.functional as F`: Provides functional neural network primitives like `F.linear` (matrix multiplication) and `F.cross_entropy` (loss calculation).
- `from transformers import AutoModelForCausalLM, AutoTokenizer`: Automatically loads the model architecture and tokenizer files using the model name string.
- `import pandas as pd`: Imports Pandas for data tables and CSV exports.

```python
print("=" * 65)
print("             STAGE 1: RUNTIME ENVIRONMENT DIAGNOSTICS")
print("=" * 65)
print(f"Python Version        : {sys.version.split()[0]}")
print(f"PyTorch Version       : {torch.__version__}")
print(f"Transformers Version  : {transformers.__version__}")
```
- Prints formatted banner text and system library versions. `sys.version.split()[0]` extracts the version number (e.g., `3.10.12`).

```python
vm = psutil.virtual_memory()
print(f"System RAM Total      : {vm.total / 1024**3:.2f} GB")
print(f"System RAM Available  : {vm.available / 1024**3:.2f} GB")
disk = shutil.disk_usage("/")
print(f"Disk Space Free       : {disk.free / 1024**3:.2f} GB")
```
- `psutil.virtual_memory()`: Queries the host computer's RAM.
- `1024**3`: Divides raw byte count by $1024^3$ ($1,073,741,824$) to convert bytes to Gigabytes (GB).
- `shutil.disk_usage("/")`: Checks how much free disk space is available for saving models.

```python
cuda_available = torch.cuda.is_available()
print(f"CUDA (GPU) Available  : {cuda_available}")
if cuda_available:
    print(f"GPU Device Name       : {torch.cuda.get_device_name(0)}")
    print(f"GPU VRAM Total        : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"BF16 Supported        : {torch.cuda.is_bf16_supported()}")
else:
    print("Running on CPU. A GPU runtime is recommended (Runtime > Change runtime type).")
print("=" * 65)
```
- `torch.cuda.is_available()`: Checks if an NVIDIA GPU is available.
- `torch.cuda.get_device_name(0)`: Gets the GPU model name (e.g. `Tesla T4`, `NVIDIA L4`, or `RTX 4090`).
- `torch.cuda.is_bf16_supported()`: Checks if hardware natively supports Brain Floating Point (BF16), which prevents numerical underflow.

---

### Cell 4: Loading the Baseline Model & Prompt Test

#### What this cell does:
Loads the unquantized `Qwen/Qwen3-0.6B` baseline model in 16-bit floating point, calculates total parameters and baseline size in Megabytes, and tests greedy text generation on a sample prompt.

```python
MODEL_ID = "Qwen/Qwen3-0.6B"
device = "cuda" if torch.cuda.is_available() else "cpu"
```
- `MODEL_ID`: The Hugging Face model repository identifier.
- `device`: Targets `"cuda"` if a GPU is detected, otherwise `"cpu"`.

```python
if cuda_available and torch.cuda.is_bf16_supported():
    dtype, BASE_NAME = torch.bfloat16, "BF16"
elif cuda_available:
    dtype, BASE_NAME = torch.float16, "FP16"
else:
    dtype, BASE_NAME = torch.bfloat16, "BF16"

MB = 1e6
```
- Selects the baseline precision: `bfloat16` if supported by hardware, otherwise `float16`.
- `MB = 1e6`: Sets 1 Megabyte to $1,000,000$ bytes (decimal MB standard used in project requirements).

```python
print(f"Loading baseline '{MODEL_ID}' on {device} ({BASE_NAME})...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
baseline_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype, device_map=device).eval()
```
- `AutoTokenizer.from_pretrained(...)`: Downloads the tokenizer vocabulary and BPE merge files.
- `AutoModelForCausalLM.from_pretrained(...)`: Downloads model weights and loads them into GPU memory with the selected precision (`torch_dtype=dtype`).
- `.eval()`: Puts the model in evaluation mode (disables dropout).

```python
n_params = sum(p.numel() for p in baseline_model.parameters())
FP32_BYTES = n_params * 4

print(f"Total parameters       : {n_params:,}")
print(f"FP32 size (params x 4) : {FP32_BYTES / MB:,.1f} MB")
print(f"{BASE_NAME} size (params x 2) : {n_params * 2 / MB:,.1f} MB")
print(f"Tied embeddings/output : {baseline_model.config.tie_word_embeddings}")
```
- `p.numel()`: Returns the number of elements in tensor `p`.
- `sum(...)`: Computes total model parameters (~596 million parameters). Note that because word embeddings are tied to the language model output head (`tie_word_embeddings = True`), their shared weights are counted once.
- `FP32_BYTES = n_params * 4`: In standard 32-bit float, each parameter uses 4 bytes (~2,386 MB).
- In 16-bit float (`BASE_NAME`), each parameter uses 2 bytes (~1,193 MB).

```python
msg = [{"role": "user", "content": "Explain why the sky is blue in three simple sentences."}]
try:
    text = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
except TypeError:
    text = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

inputs = tokenizer(text, return_tensors="pt").to(device)
with torch.no_grad():
    out = baseline_model.generate(**inputs, max_new_tokens=64, do_sample=False)

print("\nBaseline response:\n" + tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip())
```
- `tokenizer.apply_chat_template(...)`: Wraps the question with the dialogue prompt format required by Qwen3 (`<|im_start|>user ... <|im_end|>`).
- `return_tensors="pt"`: Returns PyTorch tensors.
- `with torch.no_grad():`: Disables gradient tracking during inference to conserve memory.
- `do_sample=False`: Greedy decoding (always chooses the most probable next token for deterministic evaluation).
- `out[0][inputs.input_ids.shape[1]:]`: Slices off the input prompt so only the newly generated response is decoded and printed.

---

### Cell 6: Core Quantization Math Functions (`quantizer_core.py` Part 1)

#### What this cell does:
Uses `%%writefile quantizer_core.py` to write the first part of the standalone quantization library to disk: computing quantization ranges, scale/zero-point parameters, tensor quantization/dequantization, error calculation, and 2D matrix quantization.

```python
%%writefile quantizer_core.py
```
- Jupyter notebook magic that intercepts the cell contents and writes them directly into `quantizer_core.py`.

```python
import os
import json
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
```
- Standard imports used throughout the core quantization module.

```python
def get_qrange(bits, signed=True):
    if signed:
        return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    return 0, 2 ** bits - 1
```
- Calculates integer limits $[q_{min}, q_{max}]$.
- If `bits=8` and `signed=True`:
  - $q_{min} = -2^7 = -128$
  - $q_{max} = 2^7 - 1 = 127$
- If `bits=4` and `signed=True`:
  - $q_{min} = -2^3 = -8$
  - $q_{max} = 2^3 - 1 = 7$

```python
def calculate_qparams(w_min, w_max, q_min, q_max, symmetric=False,
                      scale_dtype=None, min_scale=1e-8):
```
- Takes minimum/maximum weight bounds and computes scale ($S$) and zero-point ($Z$).

```python
    if symmetric:
        max_abs = torch.maximum(torch.abs(w_min), torch.abs(w_max))
        scale = max_abs / float(q_max)
```
- In symmetric mode, scale is determined by the maximum absolute magnitude: $S = \frac{\max(|w_{min}|, |w_{max}|)}{q_{max}}$.

```python
    else:
        w_min = torch.clamp(w_min, max=0.0)
        w_max = torch.clamp(w_max, min=0.0)
        scale = (w_max - w_min) / float(q_max - q_min)
```
- In asymmetric mode:
  - `torch.clamp(w_min, max=0.0)` ensures $w_{min} \le 0$.
  - `torch.clamp(w_max, min=0.0)` ensures $w_{max} \ge 0$.
  - This guarantees real-world $0.0$ is inside $[w_{min}, w_{max}]$ so zero padding is represented exactly.
  - Scale $S = \frac{w_{max} - w_{min}}{q_{max} - q_{min}}$.

```python
    scale = torch.clamp(scale, min=min_scale)
    if scale_dtype is not None:
        scale = scale.to(scale_dtype).to(w_min.dtype)
```
- `torch.clamp(scale, min=min_scale)`: Protects against division-by-zero if all weights are zero or constant.
- `scale.to(scale_dtype)`: **Key detail**: Scales are stored on disk as `float16`. Rounding $S$ to `float16` *before* computing integer zero points prevents floating-point mismatch when reloading the model later.

```python
    if symmetric:
        zero_point = torch.zeros_like(scale, dtype=torch.int32)
    else:
        zero_point = torch.clamp(torch.round(q_min - w_min / scale), q_min, q_max).to(torch.int32)

    return scale, zero_point
```
- In symmetric mode, $Z = 0$.
- In asymmetric mode, $Z = \text{clamp}\left(\text{round}\left(q_{min} - \frac{w_{min}}{S}\right), q_{min}, q_{max}\right)$.

```python
def quantize_tensor(tensor, scale, zero_point, q_min, q_max, target_dtype=torch.int8):
    scaled = tensor / scale + zero_point
    rounded = torch.round(scaled)
    clamped = torch.clamp(rounded, q_min, q_max)
    return clamped.to(target_dtype)
```
- `tensor / scale + zero_point`: Scales floating-point values into integer units and shifts by $Z$.
- `torch.round(...)`: Rounds to the nearest integer (round-half-to-even).
- `torch.clamp(...)`: Prevents numbers from overflowing outside $[q_{min}, q_{max}]$.
- `.to(target_dtype)`: Casts from float to integer `int8`.

```python
def dequantize_tensor(q, scale, zero_point, target_dtype=torch.float32):
    return scale.to(target_dtype) * (q.to(target_dtype) - zero_point.to(target_dtype))
```
- Formula: $\hat{x} = S \cdot (q - Z)$.
- Subtraction $(q - Z)$ is performed in floating point before multiplication by $S$.

```python
def calculate_error(original, reconstructed):
    diff = original.to(torch.float32) - reconstructed.to(torch.float32)
    return {
        "mae": torch.mean(torch.abs(diff)).item(),
        "rmse": torch.sqrt(torch.mean(diff ** 2)).item()
    }
```
- Measures reconstruction error:
  - **MAE (Mean Absolute Error)**: $\frac{1}{N} \sum |x - \hat{x}|$.
  - **RMSE (Root Mean Squared Error)**: $\sqrt{\frac{1}{N} \sum (x - \hat{x})^2}$.

```python
def quantize_weight(w, bits, granularity, group_size=128, symmetric=True,
                    scale_dtype=torch.float16, min_scale=2.0 ** -14):
    assert w.dtype == torch.float32, "cast weights to float32 BEFORE quantizing"
    out_f, in_f = w.shape
    q_min, q_max = get_qrange(bits)

    if granularity == "tensor":
        wg = w.reshape(1, 1, out_f * in_f)
    elif granularity == "channel":
        wg = w.reshape(out_f, 1, in_f)
    elif granularity == "group":
        assert in_f % group_size == 0, f"in_features {in_f} not divisible by group_size {group_size}"
        wg = w.reshape(out_f, in_f // group_size, group_size)
    else:
        raise ValueError(f"unknown granularity {granularity}")
```
- Takes a 2D weight matrix $[out\_features, in\_features]$ and reshapes it to 3D $[dim0, dim1, reduction\_dim]$:
  - `"tensor"`: `[1, 1, out_f * in_f]` (one global group).
  - `"channel"`: `[out_f, 1, in_f]` (one group per row).
  - `"group"`: `[out_f, in_f // 128, 128]` (blocks of 128 weights).

```python
    w_min = torch.min(wg, dim=-1, keepdim=True).values
    w_max = torch.max(wg, dim=-1, keepdim=True).values
    scale, zero_point = calculate_qparams(
        w_min, w_max, q_min, q_max, symmetric,
        scale_dtype=scale_dtype, min_scale=min_scale
    )
    q = quantize_tensor(wg, scale, zero_point, q_min, q_max, torch.int8).reshape(out_f, in_f)

    return {
        "q": q,
        "scale": scale.squeeze(-1),
        "zero_point": zero_point.squeeze(-1),
        "w_min": w_min,
        "w_max": w_max,
        "q_min": q_min,
        "q_max": q_max
    }
```
- Finds the min/max along the last dimension (`dim=-1`), computes scales and zero points, quantizes the weights, reshapes back to $[out\_features, in\_features]$, and returns the dictionary of parameters.

---

### Cell 7: Mathematical Unit Testing & Formula Verification

#### What this cell does:
Reloads `quantizer_core.py` and runs unit tests on tiny test tensors to mathematically verify scale, zero point, clamping, and exact zero preservation.

```python
import importlib
import quantizer_core
importlib.reload(quantizer_core)
from quantizer_core import *
```
- `importlib.reload(quantizer_core)`: Forces Python to reload the module freshly written in Cell 6.

```python
def show_math(title, values, bits, symmetric):
    w = torch.tensor(values, dtype=torch.float32)
    q_min, q_max = get_qrange(bits)
    w_min, w_max = torch.min(w), torch.max(w)
    s, z = calculate_qparams(w_min, w_max, q_min, q_max, symmetric)
    scaled = w / s
    shifted = scaled + z
    q = quantize_tensor(w, s, z, q_min, q_max)
    x_hat = dequantize_tensor(q, s, z)
    err = calculate_error(w, x_hat)
    ...
    return s, z, q, x_hat
```
- Runs every step explicitly and prints the intermediate tensors.

```python
DEMO = [-0.60, 0.00, 0.25, 0.90]
show_math("Tiny tensor", DEMO, 8, symmetric=False)
show_math("Tiny tensor", DEMO, 8, symmetric=True)
show_math("Tiny tensor", DEMO, 4, symmetric=False)
```
- Runs tests on sample values in 8-bit asymmetric, 8-bit symmetric, and 4-bit asymmetric modes.

```python
s, z, q, x_hat = show_math("REFERENCE BENCHMARK", [-1.0, 0.0, 2.0, 10.0], 8, symmetric=False)
ok = (abs(s.item() - 0.0431) < 5e-5) and z.item() == -105 and q.tolist() == [-128, -105, -59, 127] \
     and x_hat[1].item() == 0.0 and abs(x_hat[2].item() - 1.984) < 1e-3
print(f"\nVerification passed (scale 0.0431, zero point -105, stored -128/-105/-59/127, exact zero preserved): {ok}")
s_sym, _ = calculate_qparams(torch.tensor(-1.0), torch.tensor(10.0), -128, 127, symmetric=True)
print(f"Symmetric scale for reference data: {s_sym.item():.4f}")
```
- Mathematical proof test:
  - Values: $[-1.0, 0.0, 2.0, 10.0]$
  - Scale: $S = \frac{10 - (-1)}{127 - (-128)} = \frac{11}{255} \approx 0.043137$.
  - Zero Point: $Z = \text{round}\left(-128 - \frac{-1.0}{0.043137}\right) = -105$.
  - Storing real $0.0$: $q = \text{round}\left(\frac{0.0}{S} - 105\right) = -105$.
  - Reading back $0.0$: $\hat{x} = S \cdot (-105 - (-105)) = \mathbf{0.000000}$ (Exact zero!).

---

### Cell 9: INT4 Bit-Packing Engine (`quantizer_core.py` Part 2)

#### What this cell does:
Appends `pack_int4` and `unpack_int4` to `quantizer_core.py` using `%%writefile -a`. Packs two 4-bit numbers (nibbles) into a single 8-bit byte (`uint8`).

```python
%%writefile -a quantizer_core.py
```
- `-a`: Appends code to the existing file without overwriting previous functions.

```python
def pack_int4(q):
    assert q.shape[1] % 2 == 0, "number of columns must be even"
    assert int(q.min()) >= -8 and int(q.max()) <= 7, "values must fit in signed 4 bits"
    u = q.to(torch.int32) + 8
    low, high = u[:, 0::2], u[:, 1::2]
    return (low | (high << 4)).to(torch.uint8)
```
- `u = q.to(torch.int32) + 8`:
  - Signed INT4 values are in $[-8, 7]$.
  - Bitwise operations require non-negative numbers. Adding $+8$ shifts the values into unsigned nibbles $[0, 15]$ (binary `0000` to `1111`).
  - Example: $-8 + 8 = 0$ (`0000`), $0 + 8 = 8$ (`1000`), $7 + 8 = 15$ (`1111`).
- `low, high = u[:, 0::2], u[:, 1::2]`:
  - Slices even columns into `low` and odd columns into `high`.
- `high << 4`:
  - Shifts the odd number 4 bits to the left into bits 4..7.
- `low | (high << 4)`:
  - Bitwise OR merges the low 4 bits and high 4 bits into one 8-bit byte.
  - `.to(torch.uint8)`: Stores as unsigned 8-bit integer, halving the memory footprint.

```python
def unpack_int4(packed, n_cols):
    p = packed.to(torch.int32)
    low = (p & 0x0F) - 8
    high = ((p >> 4) & 0x0F) - 8
    return torch.stack([low, high], dim=-1).reshape(packed.shape[0], n_cols).to(torch.int8)
```
- `(p & 0x0F) - 8`:
  - `0x0F` in binary is `0000 1111`. Bitwise AND isolates the lower 4 bits. Subtracting 8 returns the original $[-8, 7]$ signed number.
- `((p >> 4) & 0x0F) - 8`:
  - `>> 4` shifts the upper 4 bits right into the lower positions. Masking with `0x0F` and subtracting 8 recovers the second signed number.
- `torch.stack([low, high], dim=-1).reshape(...)`: Interleaves even and odd columns back into the original 2D tensor shape.

---

### Cell 10: Bit-Packing Verification Tests

#### What this cell does:
Validates that random 4-bit integer matrices can be packed into bytes and unpacked with 100% bit-exact equivalence.

```python
importlib.reload(quantizer_core)
from quantizer_core import *

test_mat = torch.randint(-8, 8, (4, 8), dtype=torch.int8)
packed = pack_int4(test_mat)
assert torch.equal(unpack_int4(packed, 8), test_mat), "Bit packing failed!"

print("Pack/unpack round trip is bit-exact.")
print(f"Unpacked : {test_mat.numel()} values, {test_mat.numel() * test_mat.element_size()} bytes (int8 storage)")
print(f"Packed   : {packed.numel()} bytes ({packed.dtype}) -> {packed.numel() / test_mat.numel():.2f} bytes per value")
```
- Generates a random $4 \times 8$ matrix of values in $[-8, 7]$.
- Packs into a $4 \times 4$ byte matrix.
- `torch.equal(...)` verifies that every single unpacked integer matches the original input.
- Displays bytes-per-value: exactly **0.50 bytes (4 bits)** per parameter.

---

### Cell 12: `QuantizedLinear` Layer & Model Transformation (`quantizer_core.py` Part 3)

#### What this cell does:
Appends the custom `QuantizedLinear` class (drop-in replacement for `torch.nn.Linear`) and the full-model transformation logic (`quantize_model`) to `quantizer_core.py`.

```python
%%writefile -a quantizer_core.py


class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bits, granularity,
                 group_size=128, symmetric=True, has_bias=False):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.bits, self.granularity = bits, granularity
        self.group_size, self.symmetric = group_size, symmetric

        n_groups = in_features // group_size if granularity == "group" else 1
        s0 = 1 if granularity == "tensor" else out_features
```
- Sets dimensions and determines how many scale factors are needed:
  - If group quantization: $n\_groups = in\_features // 128$.
  - If per-tensor: $s0 = 1$; if per-channel/group: $s0 = out\_features$.

```python
        if bits == 8:
            self.register_buffer("weight_q", torch.zeros(out_features, in_features, dtype=torch.int8))
        elif bits == 4:
            self.register_buffer("weight_q", torch.zeros(out_features, in_features // 2, dtype=torch.uint8))
        else:
            raise ValueError("only 8-bit and 4-bit storage implemented")

        self.register_buffer("scale", torch.ones(s0, n_groups, dtype=torch.float16))

        if symmetric:
            self.register_buffer("zero_point", None)
        else:
            self.register_buffer("zero_point", torch.zeros(s0, n_groups, dtype=torch.int8))

        if has_bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.bfloat16))
        else:
            self.register_buffer("bias", None)
```
- `register_buffer(...)`: Registers persistent state tensors that are not trained by gradients, but are automatically saved in checkpoints and moved to the GPU.
- For 4-bit, columns are halved (`in_features // 2`) because two weights are packed per byte.
- In symmetric mode, `zero_point` is set to `None` to save memory.

```python
    @classmethod
    def from_float(cls, linear, bits, granularity, group_size=128, symmetric=True):
        w32 = linear.weight.data.to(torch.float32)
        r = quantize_weight(w32, bits, granularity, group_size, symmetric)
        layer = cls(linear.in_features, linear.out_features, bits, granularity,
                    group_size, symmetric, linear.bias is not None).to(linear.weight.device)
        layer.weight_q.copy_(pack_int4(r["q"]) if bits == 4 else r["q"])
        layer.scale.copy_(r["scale"].to(torch.float16))
        if not symmetric:
            layer.zero_point.copy_(r["zero_point"].to(torch.int8))
        if linear.bias is not None:
            layer.bias.copy_(linear.bias.data.to(layer.bias.dtype))
        return layer
```
- Converts a standard `nn.Linear` layer:
  1. Casts weights to `float32`.
  2. Runs `quantize_weight`.
  3. Packs into INT4 if `bits == 4`.
  4. Copies scales, zero-points, and optional biases into the new `QuantizedLinear` layer.

```python
    def dequantize_weight(self, dtype=torch.float32):
        q = unpack_int4(self.weight_q, self.in_features) if self.bits == 4 else self.weight_q
        n_groups = self.scale.shape[1]
        qg = q.to(torch.float32).reshape(self.out_features, n_groups, self.in_features // n_groups)
        if self.zero_point is not None:
            qg = qg - self.zero_point.to(torch.float32).unsqueeze(-1)
        w = qg * self.scale.to(torch.float32).unsqueeze(-1)
        return w.reshape(self.out_features, self.in_features).to(dtype)
```
- Reconstructs floating-point weights during the forward pass:
  - Unpacks INT4 if needed.
  - Subtracts `zero_point` (if asymmetric).
  - Multiplies by `scale`.
  - Reshapes back to $[out\_features, in\_features]$.

```python
    def forward(self, x):
        w = self.dequantize_weight(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, b)
```
- Runs the forward pass: dequantizes weights to match activation precision (`x.dtype`), and calls PyTorch's native `F.linear(x, w, b)`.

```python
TARGET_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

def _target_linears(model):
    return [(n, m) for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and n.split(".")[-1] in TARGET_PROJECTIONS]

def _replace(model, full_name, new_module):
    parent_name, _, child_name = full_name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child_name, new_module)

def quantize_model(base_model, cfg):
    q_model = copy.deepcopy(base_model)
    for name, lin in _target_linears(q_model):
        _replace(q_model, name, QuantizedLinear.from_float(
            lin, cfg["bits"], cfg["granularity"], cfg.get("group_size", 128), cfg["symmetric"]))
    return q_model
```
- `TARGET_PROJECTIONS`: The 7 linear projections inside each transformer block (Attention: `q, k, v, o`; MLP: `gate, up, down`). Across 28 blocks, this is $7 \times 28 = 196$ matrices.
- `_target_linears`: Finds these layers while ignoring sensitive layers (embeddings and normalizations).
- `_replace`: Dynamically swaps the standard linear layer for `QuantizedLinear`.
- `quantize_model`: Deep-copies the baseline and converts all 196 target matrices according to `cfg`.

---

### Cell 13: Memory Measurement & Standalone SafeTensors Checkpointing (`quantizer_core.py` Part 4)

#### What this cell does:
Appends memory diagnostic utilities (`tensor_bytes`, `linear_storage_bytes`), reconstruction error computation, and standalone SafeTensors export/import functions to `quantizer_core.py`.

```python
%%writefile -a quantizer_core.py


def tensor_bytes(model, only_modules=None):
    seen, total = set(), 0
    mods = only_modules if only_modules is not None else [model]
    for mod in mods:
        for t in list(mod.parameters()) + list(mod.buffers()):
            key = (t.untyped_storage().data_ptr(), t.device)
            if key in seen:
                continue
            seen.add(key)
            total += t.numel() * t.element_size()
    return total
```
- Accurately counts physical memory usage:
  - `t.untyped_storage().data_ptr()`: Gets the raw memory address pointer of the tensor storage.
  - If two layers share the same memory (like tied embeddings), `seen.add(key)` ensures the shared buffer is counted only once!

```python
def linear_storage_bytes(model):
    mods = [m for n, m in model.named_modules()
            if n.split(".")[-1] in TARGET_PROJECTIONS and isinstance(m, (nn.Linear, QuantizedLinear))]
    return tensor_bytes(model, mods), len(mods)
```
- Measures the memory of only the 196 target projection matrices.

```python
@torch.no_grad()
def compute_model_error(orig_model, q_model):
    orig = dict(orig_model.named_modules())
    abs_sum = sq_sum = 0.0
    n = 0
    for name, ql in q_model.named_modules():
        if isinstance(ql, QuantizedLinear):
            w0 = orig[name].weight.data.to(torch.float32)
            diff = w0 - ql.dequantize_weight(torch.float32)
            abs_sum += diff.abs().sum().item()
            sq_sum += (diff ** 2).sum().item()
            n += w0.numel()
    return abs_sum / n, (sq_sum / n) ** 0.5
```
- Computes global weight reconstruction error across all 196 quantized matrices.

```python
def export_checkpoint(model, out_dir, cfg=None, base_model_id=None, tokenizer=None):
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    sd = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    if getattr(model.config, "tie_word_embeddings", False):
        sd.pop("lm_head.weight", None)
    path = os.path.join(out_dir, "model.safetensors")
    save_file(sd, path, metadata={"format": "pt"})
    model.config.save_pretrained(out_dir)
    if cfg is not None:
        with open(os.path.join(out_dir, "quant_config.json"), "w") as f:
            json.dump({
                "format_version": 1,
                "base_model_id": base_model_id,
                "quantization": cfg,
                "target_projections": list(TARGET_PROJECTIONS)
            }, f, indent=2)
    if tokenizer is not None:
        tokenizer.save_pretrained(out_dir)
    return path
```
- Exports the quantized checkpoint:
  - `sd.pop("lm_head.weight", None)`: Drops duplicate tied embedding weights from the saved file.
  - `save_file(sd, ...)`: Saves weights in SafeTensors format.
  - `quant_config.json`: Stores quantization metadata (bit-width, granularity, symmetry).
  - Saves model configuration and tokenizer files.

```python
def load_quantized_model(path, device="cpu", dtype=torch.bfloat16):
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM
    with open(os.path.join(path, "quant_config.json")) as f:
        meta = json.load(f)
    cfg = meta["quantization"]
    config = AutoConfig.from_pretrained(path)
    model = AutoModelForCausalLM.from_config(config).to(dtype)
    for name, lin in _target_linears(model):
        _replace(model, name, QuantizedLinear(
            lin.in_features, lin.out_features, cfg["bits"], cfg["granularity"],
            cfg.get("group_size", 128), cfg["symmetric"], lin.bias is not None))
    sd = load_file(os.path.join(path, "model.safetensors"))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if getattr(config, "tie_word_embeddings", False):
        model.tie_weights()
        missing = [k for k in missing if k != "lm_head.weight"]
    assert not unexpected and not missing, f"load problem: missing={missing}, unexpected={unexpected}"
    return model.to(device).eval()
```
- Standalone loader: Loads the empty model architecture skeleton, swaps in empty `QuantizedLinear` layers, loads weights from `model.safetensors`, ties weights, and verifies zero missing parameters.

---

### Cell 14: Inspecting Real Weights Across Configurations A through F

#### What this cell does:
Applies configurations A through F to a real transformer layer (`layer 0 q_proj`) to observe parameter counts, scale values, and integer distributions.

```python
CONFIGS = {
    "A: INT8 per-tensor (sym)":   dict(bits=8, granularity="tensor",  symmetric=True),
    "B: INT8 per-channel (sym)":  dict(bits=8, granularity="channel", symmetric=True),
    "C: INT4 per-channel (sym)":  dict(bits=4, granularity="channel", symmetric=True),
    "D: INT4 per-channel (asym)": dict(bits=4, granularity="channel", symmetric=False),
    "E: INT4 g128 (sym)":         dict(bits=4, granularity="group", group_size=128, symmetric=True),
    "F: INT4 g128 (asym)":        dict(bits=4, granularity="group", group_size=128, symmetric=False),
}
```
- Defines the 6 quantization configurations.

```python
sample = baseline_model.model.layers[0].self_attn.q_proj
w32 = sample.weight.data.to(torch.float32)
```
- Selects the query projection matrix of layer 0 ($1024 \times 1024 = 1,048,576$ weights).

```python
rows = []
for name, cfg in CONFIGS.items():
    r = quantize_weight(w32, cfg["bits"], cfg["granularity"], cfg.get("group_size", 128), cfg["symmetric"])
    rows.append({ ... })
print(pd.DataFrame(rows).to_string(index=False))
```
- Quantizes the layer across all configurations and tabulates the scale count:
  - Per-tensor: 1 scale factor.
  - Per-channel: 1,024 scale factors.
  - Group-128: 8,192 scale factors.

---

### Cell 16: Evaluation Benchmarking Suite (Perplexity, Speed, Task Sanity)

#### What this cell does:
Defines the evaluation benchmark functions: dataset loading, perplexity calculation on WikiText-2, text generation, generation speed (tokens/sec), and sanity checks.

```python
SIZE_TARGET = 0.60
MAX_PPL_INCREASE = 0.10
USABLE_RAM_BYTES = 8e9

N_EVAL_TOKENS = 32768
PPL_WINDOW = 512
GEN_TOKENS = 128
SPEED_TOKENS = 64
```
- Sets the evaluation constraints:
  - `SIZE_TARGET`: $\ge 60\%$ size reduction vs FP32.
  - `MAX_PPL_INCREASE`: $\le 10\%$ perplexity increase vs baseline.
  - `USABLE_RAM_BYTES`: $8 \text{ GB}$ usable RAM threshold.

```python
def load_eval_ids(tok, n_tokens):
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(ds["text"])[:600000]
        source = "WikiText-2 (raw) test split"
    except Exception as e:
        print(f"WARNING: Could not load WikiText-2 ({type(e).__name__}). Using built-in fallback evaluation text.")
        text, source = FALLBACK_TEXT * 40, "built-in fallback text"
    return tok(text, return_tensors="pt").input_ids[:, :n_tokens], source

eval_ids, eval_source = load_eval_ids(tokenizer, N_EVAL_TOKENS)
```
- Loads the test split of WikiText-2 and tokenizes the first 32,768 tokens.

```python
@torch.no_grad()
def perplexity(model, ids, window=PPL_WINDOW):
    model.eval()
    nll, count = 0.0, 0
    for i in range(0, ids.shape[1] - 1, window):
        chunk = ids[:, i:i + window].to(device)
        if chunk.shape[1] < 2:
            break
        logits = model(chunk).logits
        nll += F.cross_entropy(logits[0, :-1].float(), chunk[0, 1:], reduction="sum").item()
        count += chunk.shape[1] - 1
    return math.exp(nll / count)
```
- Computes language perplexity:
  - Takes 512-token chunks.
  - Compares token predictions at $t$ with ground truth at $t+1$ using cross-entropy loss.
  - Returns $\text{PPL} = \exp\left(\frac{\text{total negative log-likelihood}}{\text{total tokens}}\right)$. Lower is better!

```python
@torch.no_grad()
def benchmark_speed(model, n_new=SPEED_TOKENS, repeats=2):
    inp = chat_inputs("Describe how a bicycle works.")
    model.generate(**inp, max_new_tokens=8, do_sample=False)
    sync()
    t0 = time.perf_counter()
    for _ in range(repeats):
        model.generate(**inp, max_new_tokens=n_new, min_new_tokens=n_new, do_sample=False)
    sync()
    return n_new * repeats / (time.perf_counter() - t0)
```
- Measures generation speed (tokens per second) using high-precision timers and `torch.cuda.synchronize()`.

```python
def task_checks(responses):
    return int("150" in responses[1]) + int("def " in responses[2])
```
- Sanity checks: verifies math reasoning ($60 \times 2.5 = 150$) and Python coding capability (`def `).

---

### Cell 18: Running the Benchmark Sweep

#### What this cell does:
Evaluates the baseline model, loops through configurations A through F, quantizes the model, records metrics, and frees GPU memory.

```python
results = []
results.append(evaluate_config(f"Baseline ({BASE_NAME})", baseline_model, None))
```
- Evaluates the unquantized baseline first.

```python
for name, cfg in CONFIGS.items():
    print(f"\nQuantizing and evaluating {name}...")
    t0 = time.time()
    qm = qz.quantize_model(baseline_model, cfg)
    build_s = time.time() - t0
    
    r = evaluate_config(name, qm, cfg)
    r["quantize_s"] = build_s
    results.append(r)
    
    del qm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```
- Builds each quantized model.
- Evaluates memory, error, perplexity, and speed.
- `del qm`, `gc.collect()`, and `torch.cuda.empty_cache()`: Reclaims memory between configurations to prevent out-of-memory crashes.

---

### Cell 20: Results Analysis, Comparison Tables & CSV Export

#### What this cell does:
Calculates reduction percentages, checks constraint compliance, prints formatted Markdown comparison tables, and saves `results.csv`.

```python
base = results[0]

for r in results:
    r["red_fp32"] = 1 - r["total_bytes"] / FP32_BYTES
    r["red_base"] = 1 - r["total_bytes"] / base["total_bytes"]
    r["lin_red"] = 1 - r["lin_bytes"] / base["lin_bytes"]
    r["bits_per_w"] = r["lin_bytes"] * 8 / LINEAR_PARAMS
    r["ppl_inc"] = r["ppl"] / base["ppl"] - 1
    r["fits_ram"] = r["total_bytes"] < USABLE_RAM_BYTES
    r["meets_size"] = r["red_fp32"] >= SIZE_TARGET
    r["meets_quality"] = r["ppl_inc"] <= MAX_PPL_INCREASE
```
- `red_fp32`: Percent reduction versus 32-bit floating point.
- `red_base`: Percent reduction versus 16-bit baseline.
- `ppl_inc`: Perplexity change: $\frac{\text{PPL}_{quant} - \text{PPL}_{base}}{\text{PPL}_{base}}$.
- `meets_size` / `meets_quality`: Checks compliance with the $\ge 60\%$ size reduction and $\le 10\%$ perplexity increase thresholds.

```python
pd.concat([size_df, quality_df.drop(columns="Config")], axis=1).to_csv("results.csv", index=False)
```
- Exports the consolidated benchmark results to a CSV file.

---

### Cell 22: Selecting the Winning Configuration

#### What this cell does:
Applies the engineering decision rules to pick the optimal model configuration.

```python
cands = [r for r in results[1:] if r["meets_size"]]
good = [r for r in cands if r["meets_quality"]]

if good:
    smallest = min(r["total_bytes"] for r in good)
    near = [r for r in good if r["total_bytes"] <= smallest * 1.01]
    winner = min(near, key=lambda r: r["ppl"])
    why = (f"Smallest configuration meeting both size and perplexity constraints "
           f"({len(near)} configuration(s) within 1% size tie-break; lowest perplexity selected)")
elif cands:
    winner = min(cands, key=lambda r: r["ppl"])
    why = "No configuration met the strict perplexity threshold; selected lowest perplexity meeting size target"
else:
    winner = max(results[1:], key=lambda r: r["red_fp32"])
    why = "Warning: No configuration met the size threshold; selected largest reduction configuration"
```
- Filters configurations meeting the size target ($\ge 60\%$) and quality bound ($\le 10\%$).
- Among qualifying candidates, finds the smallest. If sizes are within 1% of each other, breaks the tie by choosing the lowest perplexity.

---

### Cell 24: Checkpoint Export, Reload & Bit-Exact Verification

#### What this cell does:
Rebuilds the winning configuration, saves all artifacts to `final_quantized_model/`, reloads the model from disk, and tests bit-exact logits equivalence.

```python
final_name, final_cfg = winner["name"], winner["cfg"]
FINAL_DIR = "final_quantized_model"

if os.path.exists(FINAL_DIR):
    shutil.rmtree(FINAL_DIR)

final_model = qz.quantize_model(baseline_model, final_cfg).eval()
qz.export_checkpoint(final_model, FINAL_DIR, cfg=final_cfg, base_model_id=MODEL_ID, tokenizer=tokenizer)
shutil.copy("quantizer_core.py", os.path.join(FINAL_DIR, "quantizer_core.py"))
```
- Rebuilds and exports the winning model to SafeTensors.
- Copies `quantizer_core.py` directly into the export folder so the checkpoint is completely self-contained.

```python
reloaded = qz.load_quantized_model(FINAL_DIR, device=device, dtype=dtype)

with torch.no_grad():
    probe = eval_ids[:, :256].to(device)
    logits_a = final_model(probe).logits
    logits_b = reloaded(probe).logits

same_logits = torch.equal(logits_a, logits_b)
max_diff = (logits_a.float() - logits_b.float()).abs().max().item()
ppl_a = perplexity(final_model, eval_ids)
ppl_b = perplexity(reloaded, eval_ids)
gen_a = generate_text(final_model, EVAL_PROMPTS[1], GEN_TOKENS)
gen_b = generate_text(reloaded, EVAL_PROMPTS[1], GEN_TOKENS)
```
- Loads the model back from disk using `load_quantized_model`.
- Feeds 256 tokens into both the in-memory model and the reloaded-from-disk model.
- `torch.equal(logits_a, logits_b)`: Confirms that output logits match identically down to the lowest bit (`max_diff = 0.00e+00`), proving that saving and reloading does not cause any precision degradation.

---

# 4. Evaluator Q&A and Presentation Defense

Be prepared to answer these technical questions during your evaluation:

### 1. "Why didn't you quantize the token embeddings and the language model head?"
> **Answer**:  
> In Qwen3-0.6B, the token embedding matrix and `lm_head` are tied and share weights. Quantizing the vocabulary projections introduces high reconstruction noise into logit probabilities, causing immediate text degradation. By quantizing only the 196 internal projection matrices (attention and MLP layers), we quantize ~75% of total parameters while keeping embeddings in 16-bit float, protecting model stability.

### 2. "Why did per-channel 4-bit (Configs C & D) fail or degrade, while Group-128 (Configs E & F) succeeded?"
> **Answer**:  
> In per-channel quantization, a single row has 1,024 or 2,048 weights mapped into only 16 integer bins. A single outlier weight inflates the scale $S$, collapsing all other weights into zero. In Group-128, every consecutive 128 weights receive their own scale and zero-point. This localizes outliers and preserves fine-grained weight distributions, keeping perplexity increase under 10%.

### 3. "Why does your quantized model run slightly slower than the unquantized baseline?"
> **Answer**:  
> Our implementation is written from scratch in raw PyTorch without custom C++/CUDA fused kernels. In our `QuantizedLinear` forward pass, weights are dynamically dequantized into float tensors before calling `F.linear`. The goal here is proving algorithmic correctness and reducing memory footprint. In production systems (like AWQ or Marlin), fused CUDA kernels perform integer unpacking and matrix multiplication simultaneously in GPU SRAM cache for speedups.

### 4. "How does INT4 bit-packing physically reduce memory?"
> **Answer**:  
> Computer architectures address memory in 8-bit bytes. Storing a 4-bit value in a standard `torch.int8` tensor still consumes 8 bits (1 full byte). Our `pack_int4` function uses bitwise operations (`(low | (high << 4))`) to pack two 4-bit values into a single unsigned 8-bit byte (`uint8`), reducing physical weight storage by exactly 50%.
