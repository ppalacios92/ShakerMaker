# Shakermaker Convolution Optimization - Stage 2

## Overview

This document describes the optimizations implemented to improve the velocity of the convolution process in Stage 2 (`run_fast`) of Shakermaker. The optimizations are designed to work seamlessly with both **progressive** and **legacy** writing modes.

## Architecture Review

### Three-Stage Pipeline

```
Stage 0 (gen_pairs)    Stage 1 (compute_gf)    Stage 2 (run_fast)
──────────────────────  ──────────────────────  ──────────────────────
• Geometry vectorization • FK kernel computation  • O(1) slot lookup
• JAA greedy algorithm   • Write tdata to HDF5    • _call_core_fast()
• HDF5 pair mapping                                  • 3x convolve() per pair
                                                       • add_to_response()
```

### Writing Modes

**Progressive Mode** (O(1) RAM):
- Writes data immediately to HDF5 after each station
- Calls `flush()` to ensure data is written to disk
- Calls `clear_response()` to release RAM after each station
- Recommended for large simulations (>1000 stations)

**Legacy Mode** (High RAM):
- Accumulates all station data in `_velocities` dictionary
- Batch processes all data in `close()` method
- Processes interpolation, differentiation (acceleration), and integration (displacement)
- Writes all data at once to HDF5

## Changes Implemented

### 1. `sourcetimefunction.py`

#### New FFT Kernel Caching

Added methods to cache the pre-computed FFT of the source time function kernel:

```python
def _get_fft_kernel(self, n_signal):
    if self._fft_kernel is None or self._fft_kernel_valid_for_dt != self._dt:
        kernel = self.data
        n_fft = _get_fft_kernel_size(n_signal, len(kernel))
        self._fft_kernel = fft(kernel, n=n_fft)
        self._fft_kernel_size = n_fft
        self._fft_kernel_valid_for_dt = self._dt
    return self._fft_kernel, self._fft_kernel_size
```

#### New Batch Convolution Methods

**`convolve_fft(val, t)`**: Single-component FFT convolution
- Uses cached FFT kernel
- Avoids redundant FFT computation for the STF

**`convolve_batch(seis, t)`**: Multi-component FFT convolution
- Accepts seismogram array with shape `(3, n_samples)` or `(n, n_samples)`
- Single FFT operation for all components
- Ideal when input and output sampling rates match

**`convolve_batch_with_resample(seis, t, t_resampled)`**: Full resample + batch FFT
- Resamples input to STF time grid (single operation for 3 components)
- Performs FFT convolution with cached kernel
- Resamples output back to original time grid
- **Primary method used in Stage 2 optimization**

### 2. `shakermaker.py` - `run_fast()`

#### Optimization 1: Move STF dt Setting Outside Loops

**Before:**
```python
for i_station in range(nstations):
    for i_psource, psource in enumerate(self._source):
        # ... inside inner loop
        psource.stf.dt = dt  # Called for EVERY source-station pair
```

**After:**
```python
# Set once, before any station loop
for psource in self._source:
    psource.stf.dt = dt

for i_station in range(nstations):
    # ...
```

**Benefit**: Eliminates redundant `_generate_data()` calls (O(nstations × nsources) → O(nsources))

#### Optimization 2: Batch by Slot (Reduce HDF5 Reads)

**Before:**
```python
for i_psource, psource in enumerate(self._source):
    k = int(pair_to_slot[i_station * nsources + i_psource])
    tdata = hfile[f"/tdata_dict/{k}_tdata"][:]  # HDF5 read EVERY iteration
```

**After:**
```python
slot_to_sources = {}
for i_psource, psource in enumerate(self._source):
    k = int(pair_to_slot[i_station * nsources + i_psource])
    if k not in slot_to_sources:
        slot_to_sources[k] = []
    slot_to_sources[k].append((i_psource, psource))

for k, source_list in slot_to_sources.items():
    tdata = hfile[f"/tdata_dict/{k}_tdata"][:]  # HDF5 read ONCE per slot
    for i_psource, psource in source_list:
        # ... process sources sharing same tdata
```

**Benefit**: Reduces HDF5 reads by factor of `slot_reuse_factor` (typically 10-100x for dense fault models)

#### Optimization 3: Batch 3-Component Convolution

**Before:**
```python
z_stf = psource.stf.convolve(z, t_arr)  # 1. Resample z
e_stf = psource.stf.convolve(e, t_arr)  # 2. Resample e
n_stf = psource.stf.convolve(n, t_arr)  # 3. Resample n
```

**After:**
```python
seis = np.stack([z, e, n])  # Stack 3 components
z_stf, e_stf, n_stf = psource.stf.convolve_batch_with_resample(
    seis, t_arr, np.arange(t_arr[0], t_arr[-1], dt))
```

**Benefit**: 
- Single resampling operation (vs 3 separate)
- Single FFT operation (vs 3 separate)
- Vectorized numpy operations

## How Optimizations Work With Both Modes

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         OPTIMIZED STAGE 2                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ PRE-LOOP (once): Set STF dt for all sources                       │   │
│  │ for psource in source: psource.stf.dt = dt                        │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              ↓                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ STATION LOOP (per station)                                        │   │
│  │                                                                    │   │
│  │  slot_to_sources = {}  # Batch by slot (reduce HDF5 reads)          │   │
│  │  for i_psource, psource:                                          │   │
│  │      k = pair_to_slot[...]                                        │   │
│  │      slot_to_sources[k].append((i_psource, psource))              │   │
│  │                                                                    │   │
│  │  for k, source_list in slot_to_sources.items():                    │   │
│  │      tdata = hfile[f"/tdata_dict/{k}_tdata"][:]  # READ ONCE     │   │
│  │                                                                    │   │
│  │      for i_psource, psource in source_list:                       │   │
│  │          z, e, n, t0 = _call_core_fast(tdata, ...)               │   │
│  │                                                                    │   │
│  │          # OPTIMIZED: Batch 3 components at once                   │   │
│  │          seis = np.stack([z, e, n])                              │   │
│  │          z_stf, e_stf, n_stf = stf.convolve_batch_with_resample()│   │
│  │                                                                    │   │
│  │          station.add_to_response(z_stf, e_stf, n_stf, ...)         │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              ↓                                          │
│         ┌────────────────────────────┬────────────────────────────┐     │
│         ↓                            ↓                            ↓     │
│  ┌──────────────┐          ┌──────────────┐          ┌──────────────┐   │
│  │ PROGRESSIVE  │          │    LEGACY    │          │  BOTH MODES  │   │
│  │              │          │              │          │              │   │
│  │ write_station│          │ accumulate  │          │ SAME result  │   │
│  │ → HDF5 disk │          │ in RAM       │          │ (identical   │   │
│  │ → flush()   │          │ → close()    │          │  seismograms)│   │
│  │ → clear()   │          │   → HDF5     │          │              │   │
│  └──────────────┘          └──────────────┘          └──────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Performance Improvements Summary

| Optimization | Benefit | Both Modes? |
|--------------|---------|-------------|
| **STF dt outside loop** | Eliminates redundant `_generate_data()` calls | ✅ Yes |
| **Batch by slot** | Reduces HDF5 reads by `slot_reuse_factor` | ✅ Yes |
| **FFT kernel cache** | Pre-computes FFT, reuses for all sources | ✅ Yes |
| **Batch 3-component** | Single resample + FFT vs 3x separate | ✅ Yes |
| **Progressive writing** | O(1) RAM, immediate disk write | ✅ Yes |
| **Legacy mode** | Works identically, just delayed write | ✅ Yes |

## Expected Performance Gains

| Scenario | Estimated Speedup |
|----------|------------------|
| Convolution (FFT kernel cache + batch) | 2-5x |
| HDF5 I/O (batch by slot) | 2-10x |
| Combined (I/O + convolution) | 5-20x |

## Memory Impact

| Mode | RAM Usage | Optimization Effect |
|------|----------|-------------------|
| **Progressive** | O(1) per station | Same O(1), faster convolution |
| **Legacy** | O(nstations × nsources) | Same O(n), faster convolution |

## Backward Compatibility

The original `convolve()` method is preserved and unchanged. All existing code using the standard convolution will continue to work without modification. The optimized methods are additive and are used automatically by `run_fast()`.

## Files Modified

1. **`shakermaker/sourcetimefunction.py`**
   - Added FFT kernel caching infrastructure
   - Added `convolve_fft()` method
   - Added `convolve_batch()` method
   - Added `convolve_batch_with_resample()` method

2. **`shakermaker/shakermaker.py`**
   - Modified `run_fast()` to use optimized convolution
   - Moved STF dt initialization outside loops (line 1127-1128)
   - Added slot batching to reduce HDF5 reads (lines 1164-1169)
   - Integrated `convolve_batch_with_resample()` for 3-component processing (lines 1192-1194)
