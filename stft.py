import numpy as np


def create_window(frame_size):
    """Create a Hann window."""
    return np.hanning(frame_size)


def get_frames(audio, frame_size, hop_size):
    """Split audio into overlapping frames."""
    frames = []

    for start in range(0, len(audio) - frame_size, hop_size):
        frame = audio[start:start + frame_size]
        frames.append(frame)

    return np.array(frames)


def apply_window(frames, window):
    """Apply the window to every frame."""
    return frames * window


def calculate_fft(frames):
    """Calculate the FFT of every frame."""
    return np.fft.rfft(frames, axis=1)


def calculatr_stft(audio, frame_size, hop_size):
    window = create_window(frame_size)
    
    frames = get_frames(
        audio,
        frame_size,
        hop_size
    )

    windowed_frames = apply_window(
        frames,
        window
    )

    spectra = calculate_fft(
        windowed_frames
    )

    return spectra


def calculate_ifft(spectra, frame_size):
    """Convert frequency-domain frames back to time-domain frames."""

    return np.fft.irfft(
        spectra,
        n=frame_size,
        axis=1
    )


def overlap_add(frames, window, hop_size):
    """Reconstruct the audio using overlap-add."""

    frame_size = frames.shape[1]

    output_length = (
        (len(frames) - 1) * hop_size
        + frame_size
    )

    output = np.zeros(output_length)
    window_sum = np.zeros(output_length)

    for i, frame in enumerate(frames):

        start = i * hop_size
        end = start + frame_size

        # Apply synthesis window
        windowed_frame = frame * window

        # Add frame to output
        output[start:end] += windowed_frame

        # Keep track of window energy
        window_sum[start:end] += window ** 2

    # Avoid division by zero
    window_sum = np.maximum(window_sum, 1e-10)

    # Normalize overlap regions
    output /= window_sum

    return output


def calculate_istft(reconstructed, frame_size, hop_size):
    window = create_window(frame_size)

    reconstructed_frames = calculate_ifft(
        reconstructed,
        frame_size
    )

    reconstructed_audio = overlap_add(
        reconstructed_frames,
        window,
        hop_size
    )

    return reconstructed_audio