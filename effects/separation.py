"""Source separation for /process/separate: a thin wrapper around the
semi-supervised NMF pipeline of main_semi_nmf.main().

The algorithm is the separation track's (semi_supervised_nmf, nmf_sep,
stft, utils), called as-is with main()'s settings. main() is a script
(fixed file paths, plots and WAVs written to disk), so its steps are
repeated here without the file I/O:

    |STFT| of the mono mix → NMF (V ≈ B · G, NUM_COMPONENTS templates)
    → score every component → group them into stems → one soft mask per
    stem (the groups' masks scaled to sum to 1) → mask × STFT → iSTFT

Keep this in step with main() if its settings or grouping change.

Stereo: the masks are computed once from the mono mix (main() analyses
the mono mix too) and applied to each channel's STFT, so every stem has
the input's channel layout. Each stem also has the input's length (see
_padding), and the stems add up to the input.
"""

import threading

import numpy as np

import nmf_sep
import semi_supervised_nmf
import stft
import utils

# main_semi_nmf.main()'s settings.
FRAME_TIME_MS = 40
OVERLAP = 0.5
NUM_COMPONENTS = 20
PERCUSSION_MIN = 2
BASS_THRESHOLD = 0.6
VOCAL_SHARE = 3 / 5  # of the components that are still free

# main()'s stem keys -> the names the app shows, in display order.
# "harmonics" is whatever pitched sound is left (guitar, keys, strings...).
STEM_NAMES = {"percussion": "percussion", "bass": "bass", "vocal": "vocals", "remaining": "harmonics"}
STEMS = tuple(STEM_NAMES.values())

# The NMF starts from random B and G; a fixed seed gives the same stems
# for the same track every time (the global RNG is restored afterwards).
SEED = 0
# One separation at a time: it is CPU- and memory-heavy, and the seed
# above lives in NumPy's global RNG.
_LOCK = threading.Lock()


def frame_sizes(sr):
    """main()'s frames: FRAME_TIME_MS rounded up to a power of two, with
    OVERLAP. (2048 / 1024 samples at 44.1 kHz.)"""
    frame_size = int(sr * FRAME_TIME_MS / 1000)
    frame_size = 1 << (frame_size - 1).bit_length()
    return frame_size, int(frame_size * (1 - OVERLAP))


def component_groups(df, num_components=NUM_COMPONENTS):
    """main()'s choice of NMF components for each stem, from the scored
    component table (one row per component, in component order)."""
    groups = {}
    groups["percussion"] = (
        df.sort_values(by="percussion_score", ascending=False).head(PERCUSSION_MIN).index.tolist())
    groups["bass"] = df[df["bass_score"] > BASS_THRESHOLD].index.tolist()

    used = sum(groups.values(), [])
    vocal_max = int((num_components - len(used)) * VOCAL_SHARE)
    groups["vocal"] = df.sort_values(by="vocal_score", ascending=False).head(vocal_max).index.tolist()

    used = sum(groups.values(), [])
    groups["remaining"] = [i for i in range(num_components) if i not in used]
    return groups


def stem_masks(mono, sr):
    """One soft mask per stem (main()'s keys), each shaped like
    stft.calculatr_stft(mono), plus the component groups."""
    frame_size, hop_size = frame_sizes(sr)

    # utils.load_audio (what main() reads the track with) scales the mono
    # mix to a peak of 1; the masks are computed from that.
    peak = np.max(np.abs(mono)) if mono.size else 0
    if peak > 0:
        mono = mono / peak

    spectra = stft.calculatr_stft(mono, frame_size, hop_size)
    magnitude = utils.calculate_magnitude(spectra)

    state = np.random.get_state()
    np.random.seed(SEED)
    try:
        B, G = semi_supervised_nmf.separate_sources_nmf(
            X=magnitude.T, num_components=NUM_COMPONENTS, diagnostics=False)
    finally:
        np.random.set_state(state)

    nmf_mag = (B @ G).T  # main()'s sum of the components' outer products

    df = nmf_sep.analyze_nmf_components(B, G, sample_rate=sr, n_fft=frame_size)
    df_vocal = nmf_sep.analyze_vocal_features(B, G, sample_rate=sr, n_fft=frame_size)
    df = nmf_sep.score_components(df.merge(df_vocal, on="component", how="left"))

    groups = component_groups(df, B.shape[1])
    masks = {
        # get_custom_mask can't take an empty group (e.g. no component
        # scores as bass); that stem is silent instead.
        key: nmf_sep.get_custom_mask(B, G, nmf_mag, comps=comps) if comps else np.zeros(spectra.shape)
        for key, comps in groups.items()
    }
    nmf_sep.get_rest_mask(masks)  # scales the masks in place so they sum to 1
    return masks, groups


def _padding(sr):
    """Zeros added before and after the track: one frame on each side.

    stft.calculatr_stft drops the samples after its last full frame, and
    the overlap-add in stft.calculate_istft divides by a window sum that
    is ~0 in the first and last few samples (a masked frame is no longer
    zero there, so those samples blow up). With a frame of silence on
    both sides, every real sample sits where the frames overlap properly;
    the padding is cut off again afterwards.
    """
    return frame_sizes(sr)[0]


def separate(audio, sr):
    """Split audio into stems.

    audio: shape (frames,) or (frames, channels).
    Returns {stem name: audio} in STEMS order, each stem float64 with the
    input's shape.
    """
    audio = np.asarray(audio, dtype=np.float64)
    channels = audio[:, None] if audio.ndim == 1 else audio
    n = len(channels)
    frame_size, hop_size = frame_sizes(sr)
    pad = _padding(sr)
    padded = np.pad(channels, [(pad, pad), (0, 0)])

    with _LOCK:
        masks, _ = stem_masks(padded.mean(axis=1), sr)

    stems = {name: np.zeros_like(channels) for name in STEMS}
    for ch in range(channels.shape[1]):
        spectra = stft.calculatr_stft(padded[:, ch], frame_size, hop_size)
        for key, mask in masks.items():
            stem = stft.calculate_istft(utils.apply_mask(spectra, mask), frame_size, hop_size)
            stem = stem[pad:pad + n]
            stems[STEM_NAMES[key]][:len(stem), ch] = stem

    if audio.ndim == 1:
        return {name: stem[:, 0] for name, stem in stems.items()}
    return stems
