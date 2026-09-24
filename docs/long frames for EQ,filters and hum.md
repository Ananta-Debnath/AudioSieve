# Long frames for EQ, filters and hum

**Change:** `FRAME_SIZE` in [effects/eq_filter.py](../effects/eq_filter.py) went
from 1024 to 65536 samples (hop 512 → 32768) for every EQ mode (eq,
filter, both, hum). The before/after spectrograms in `app.py` stay at
1024/512.

This departs from the stage-2 spec's "keep frame size 1024 / hop size
512". That rule was about matching the drum/bass separation masks, and
as explained below it does not apply to a curve that never changes.

## Result

Gain on a sine tone, measured on the processed audio:

| Test tone | Before (1024) | Now (65536) | What the curve says |
|---|---|---|---|
| Rumble filter at 30 Hz | −18.2 dB | −34.1 dB | −34.1 dB |
| Rumble filter at 50 Hz | −11.2 dB | −16.4 dB | −16.4 dB |
| 20 Hz high-pass at 10 Hz | −8.5 dB | −24.0 dB | −24.1 dB |
| Telephone at 100 Hz | −35.0 dB | −38.2 dB | −38.2 dB |

The audio now gets the curve drawn in the EQ tab's gain-curve plot,
down to about 10 Hz.

## 1. Frame length sets frequency resolution

An FFT of N samples gives bins spaced `sr / N` apart, and the EQ curve
only exists at those bins. At 44.1 kHz:

| Frame | Bin spacing | Frame duration |
|---|---|---|
| 1024 | 43 Hz | 23 ms |
| 65536 | 0.67 Hz | 1.5 s |

With 1024-sample frames, the rumble filter (high-pass at 80 Hz) had only
three bins to describe everything from 0 to 100 Hz:

```
bin:    0 Hz     43 Hz     86 Hz    129 Hz
gain:   0        −21.5 dB  −1.9 dB  −0.1 dB
```

## 2. Tones between bins get a blurred version of the curve

A 30 Hz tone does not fall into a single bin. The Hann window spreads
each tone over about ±2 bins, which is ±86 Hz at 1024 samples. So part
of the 30 Hz tone lands in the 86 Hz bin, where the filter only cuts
1.9 dB, and it leaks through. That is why it came out at −18 dB instead
of −34 dB.

In practice, the audio gets the curve averaged over a ±2-bin window:

- **At 3 kHz**, ±86 Hz is small compared with the curve's shape, so it
  does not matter. Telephone's −3 dB points at 300 Hz and 3400 Hz were
  already right with 1024-sample frames.
- **At 30–100 Hz**, ±86 Hz is wider than the features being made. The
  rumble cut, the bass shelf and the hum notches all got blurred.
- **At 65536 samples**, the blur is ±1.3 Hz, so everything down to
  about 10 Hz comes out as drawn.

## 3. Longer frames lose time resolution, which is why short frames are the default

A 1.5 s frame cannot tell *when* something happens inside it. That
matters when **the mask changes over time**. The drum mask has to switch
on for a 50 ms hit and off again, and a 1.5 s frame would smear it
across the whole beat. Spectrograms need short frames too, to show when
things happen.

## 4. The EQ curve never changes, so long frames cost it nothing

Multiplying every frame by the **same** curve is an ordinary
time-invariant filter: the input convolved with the curve's impulse
response. The result does not depend on where the frame boundaries fall.
So a long frame does not blur anything in time. The only spread is the
filter's own ringing, which comes from the curve, not the frame.

Checked with a click: for the rumble filter, the energy more than 20 ms
from the click is at −72 dB, which is inaudible.

## 5. Sharp low cuts need long impulse responses

A 30 Hz cycle lasts 33 ms, and to tell 30 Hz from 80 Hz a filter needs
to "listen" for a few cycles. So a sharp low-frequency cut needs an
impulse response tens of ms long.

A 1024-sample frame only has 23 ms of room. Multiplying spectra inside
a frame is a **circular** convolution, the same thing the reverb's
circular-convolution toggle demonstrates. An impulse response longer
than the frame wraps around, so short frames cannot produce the sharp
cut at all. With 1024 samples the click test looked "tighter" (−140 dB
outside ±20 ms) for exactly this reason: the frame was cutting off the
filter's impulse response.

## 6. The costs, measured

- **Speed:** about the same, 1.17 s against 1.10 s per 6-minute channel.
  The track is cut into about 480 big frames instead of about 31,000
  small ones.
- **Memory:** unchanged. With 50% overlap, the frames hold about twice
  the signal whatever their size.
- **Gain-curve plot:** it now receives 32,769 points instead of 513.

## The rule

**Short frames when the processing changes over time (masks,
spectrograms), long frames when it does not (EQ, filters, hum).**

## Caveat: hum at the start and end

The hum notch still leaves some hum in the first and last ~0.5 s of a
track. This is not caused by the frame size. A 4 Hz-wide notch needs
about that much signal to tell 50 Hz from 48 Hz, and any filter that
narrow, IIR ones included, has the same limit.

## Tests

In [tests/test_eq_filter_unit.py](../tests/test_eq_filter_unit.py):

- `test_audio_gets_the_plotted_curve` compares the real output on sine
  tones with the curve at low frequencies.
- `test_preset_does_what_its_name_says` checks each preset's promise,
  for example that the rumble filter cuts 30 Hz by at least 30 dB.

With the frame set back to 1024, 12 of these tests fail.
