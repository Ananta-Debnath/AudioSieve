# PIVOT STAGE 03 — Reverb (FFT Convolution) + Echo/Delay (Difference Equation)

Branch: `wasek`
Depends on: Pivot Stage 1 (Flask shell, stub routes) and Pivot Stage 2 (EQ/Filter, working).
Deadline pressure: this stage should be completable in one session.

## 0. Context and ground rules

This app is a signal-processing lab for a Signals & Systems course. The core rule is that
**every effect's computation must be written by us from signal-processing first principles.**
No black-box effect implementations.

Forbidden for the effect computation itself:
- `scipy.signal.fftconvolve`
- `scipy.signal.convolve`
- `scipy.signal.lfilter`
- `np.convolve`
- Any Web Audio / Tone.js effect node

Allowed:
- `np.fft.rfft` / `np.fft.irfft`
- Plain numpy array math
- The existing helpers in `utils.py` for I/O, normalization, and clipping protection

`np.convolve` and naive Python loops MAY be used inside **tests** as reference implementations.

Do NOT modify:
- `effects/eq_filter.py`
- `stft.py`
- Anything related to separation

Reuse existing utilities wherever possible.

## 1. Reverb — `effects/reverb.py`

Reverb is modeled as an LTI system: output = input convolved with a room impulse response (IR).
Convolution is computed via the convolution theorem, i.e. multiplication in the frequency domain.

### 1.1 `generate_impulse_response(sr, rt60, pre_delay_ms, seed=0) -> np.ndarray`

Builds a synthetic IR as exponentially decaying white noise.

- Decay length: `L = int(rt60 * sr)`, capped at 5 seconds.
- Time axis: `t = np.arange(L) / sr`.
- Envelope, chosen so amplitude falls 60 dB at `t = rt60`: `env = np.exp(-6.9078 * t / rt60)`.
  The constant 6.9078 is ln(1000).
- Noise: `noise = rng.standard_normal(L)` using `np.random.default_rng(seed)`, for reproducibility.
- Decay: `decay = noise * env`.
- Pre-delay: prepend `int(pre_delay_ms / 1000 * sr)` zeros to `decay`.
- Energy normalization, so the wet level is predictable regardless of rt60: `h = h / np.sqrt(np.sum(h**2))`.
- Return a float64 1-D array.

### 1.2 `fft_convolve(x, h) -> np.ndarray` — LINEAR convolution

- `N = len(x) + len(h) - 1`
- `nfft` = next power of two ≥ N
- `X = np.fft.rfft(x, nfft)`
- `H = np.fft.rfft(h, nfft)`
- `y = np.fft.irfft(X * H, nfft)[:N]`
- Return `y`, which has length N. The docstring must explain that zero-padding to ≥ N is what
  prevents circular wraparound.

### 1.3 `circular_convolve(x, h) -> np.ndarray` — deliberate demo of the WRONG way

- Uses FFT length `len(x)` exactly. If `len(h) > len(x)`, truncate `h` to `len(x)`.
- Returns an output of length `len(x)`.
- The reverb tail wraps around onto the start of the track. This function exists only for the
  educational toggle and must be clearly commented as such.

### 1.4 `apply_reverb(audio, sr, rt60=1.5, pre_delay_ms=20, wet=0.3, circular=False) -> np.ndarray`

- Accept mono (1-D) or multichannel audio in whatever shape `utils` load returns.
  Process each channel independently.
- Use seed = channel index, so stereo gets slightly decorrelated tails, which gives width.
- Linear mode:
  - `wet_sig = fft_convolve(x, h)`
  - `dry` = `x` zero-padded to the same length
  - `out = (1 - wet) * dry + wet * wet_sig`
  - Output is longer than the input: the tail is kept.
- Circular mode: same mix formula, but using `circular_convolve`. Output length equals the input length.
- Finish with the existing normalization / clipping protection from `utils.py`.

### 1.5 Parameter ranges

Validate in the route; clamp or return 400 with a clear message.

| Parameter      | Range     | Default |
|----------------|-----------|---------|
| `rt60` (s)     | 0.2 – 5.0 | 1.5     |
| `pre_delay_ms` | 0 – 100   | 20      |
| `wet`          | 0.0 – 1.0 | 0.3     |
| `circular`     | bool      | false   |

## 2. Echo/Delay — `effects/echo.py`

Two modes are supported, both defined by difference equations. `D` is the delay in samples and `g` is the gain.

**Feedforward (single echo, FIR):**

    y[n] = x[n] + g · x[n − D]

**Feedback (repeating decaying echoes, IIR):**

    y[n] = x[n] + g · y[n − D]

The feedback mode is stable only for |g| < 1. Clamp `g` to `[0, 0.9]`.

### 2.1 `apply_echo(audio, sr, delay_ms=350, gain=0.5, mix=0.5, mode="feedback") -> np.ndarray`

- `D = max(1, int(delay_ms / 1000 * sr))`

**Tail length.** Extend the input with zeros so echoes can ring out.
- Feedforward: tail = `D`.
- Feedback: tail = `D * K`, where `K = ceil(log(0.001) / log(g))`. This is the number of repeats
  until an echo is 60 dB down. Cap the tail at 5 seconds. If `g == 0`, the tail is 0.

**Feedforward implementation.** Vectorized slice addition:

    y = x.copy()
    y[D:] += g * x[:-D]

**Feedback implementation.** This MUST be vectorized in blocks; do not use a per-sample Python loop.
Because `y[n]` depends only on `y[n − D]`, you can process the signal in consecutive blocks of
length D:

    y[0:D] = x[0:D]
    y[kD:(k+1)D] = x[kD:(k+1)D] + g · y[(k−1)D:kD]

Handle the final partial block. This is exact, not an approximation. State this in the docstring.

**Mixing.** Mix the wet echo component with the dry signal:

    out = dry + mix · (y − dry)

At `mix = 0` the output equals the dry input. At `mix = 1` the output equals the full difference-equation output.

Handle multichannel audio the same way as reverb (per channel). Finish with the existing
normalization / clipping protection.

### 2.2 `echo_frequency_response(sr, delay_ms, gain, mode, n_points=2048) -> (freqs_hz, mag_db)`

Evaluate the frequency response on `ω ∈ [0, π]` and map it to Hz.

- Feedforward: `H(e^jω) = 1 + g · e^(−jωD)`. This is a comb filter with peaks and notches.
- Feedback: `H(e^jω) = 1 / (1 − g · e^(−jωD))`.
- Return the magnitude in dB, with a small epsilon inside the log.

For long delays the comb teeth are extremely dense. The frontend will zoom into a low-frequency
range (e.g. 0–200 Hz) to make them visible; the function itself just returns the full range.

### 2.3 Parameter ranges

| Parameter  | Range                      | Default    |
|------------|----------------------------|------------|
| `delay_ms` | 20 – 2000                  | 350        |
| `gain`     | 0.0 – 0.9                  | 0.5        |
| `mix`      | 0.0 – 1.0                  | 0.5        |
| `mode`     | `feedforward` \| `feedback` | `feedback` |

## 3. Routes (`app.py`)

**`/process/reverb` and `/process/echo`.** Replace the stubs with real processing. Follow exactly
the same request and response pattern as the working `/process/eq` route: same upload handling,
same output WAV saving, same download response. Parse and validate the parameters from sections
1.5 and 2.3.

**`GET /response/reverb?rt60=&pre_delay_ms=`** — returns JSON for the frontend to plot:

    {"t": [...seconds...], "h": [...]}

Downsample to ≤ 4000 points for plotting (simple decimation is fine).

**`GET /response/echo?delay_ms=&gain=&mode=`** — returns JSON:

    {"freqs": [...], "mag_db": [...]}

Both response routes need no upload; they depend only on the parameters.

Update the barebones test HTML so both effects can be exercised manually with their parameters,
including the circular toggle.

## 4. Tests — `tests/test_reverb_echo.py` (pytest)

1. `fft_convolve` matches `np.convolve` (reference only) on random signals of lengths
   (1000, 300) and (37, 512), within `atol = 1e-9`.
2. `circular_convolve` output length equals the input length, and its result DIFFERS from the
   first `len(x)` samples of linear convolution when `len(h) > 1`. This proves the wraparound.
3. An impulse through `apply_reverb` with `wet=1` and normalization disabled (or compare shape
   after normalization) reproduces the IR.
4. IR check: energy is 1 after normalization, and the envelope at `t = rt60` is ~60 dB below
   `t = 0`. Check the envelope, not the noise.
5. Feedforward echo on an impulse gives nonzero samples only at 0 and D, with value g at D.
6. Feedback echo on an impulse gives value `g^k` at `k·D` for k = 0..5.
7. The block-vectorized feedback output equals a naive per-sample loop reference on a short
   random signal (e.g. 5000 samples, D = 123, non-multiple length), within `1e-12`.
8. Gain above 0.9 is clamped, and the output stays finite and bounded.
9. `mix=0` returns the dry signal (echo); `wet=0` returns the dry signal plus zero-padded tail (reverb).
10. Performance: reverb and echo on 6 minutes of 44.1 kHz stereo random noise each finish in
    under ~5 s on a laptop. Mark this test `@pytest.mark.slow`.

## 5. Acceptance criteria

- All tests pass: `pytest -m "not slow"` quickly, plus the slow test once manually.
- Reverb audibly sounds like a room.
  - Short `rt60` sounds like a small room; long `rt60` sounds like a hall.
  - The circular toggle audibly smears the reverb tail onto the song's opening.
- Echo gives clearly spaced repeats.
  - Feedforward gives exactly one repeat.
  - Feedback gives decaying repeats.
- Both `/response/*` endpoints return valid JSON with the expected shapes.
- No forbidden library calls in `effects/` (see section 0).
- Commit with the message: `pivot stage 03: reverb (fft convolution) + echo (difference equation)`.

## 6. Out of scope

- Recorded/real-room IRs
- Damping filters on the reverb tail
- Ping-pong stereo delay
- Frontend styling (the next stage)
- Separation integration
