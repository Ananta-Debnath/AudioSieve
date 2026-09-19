import os
import numpy as np

import utils
import stft
import nmf
import nmf_sep
import drum_mask_gen
import vocal_mask_gen

def main():
    # -------------------------
    # Configuration
    # -------------------------
    def make_dir(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
            
    make_dir("Spectograms")
    make_dir("Audio")
    make_dir("Spectograms/nmf")
    make_dir("Audio/nmf")
    make_dir("NMF")
    make_dir("Audio/NMF/groups")

    input_file = "Audio/percussive_only.wav"
    output_file = "Audio/reconstructed.wav"
    duration = 20 # seconds

    frame_size = 1024
    hop_size = 512
    nmf_components = 10

    def spectogram_filename(name=None):
        if name:
            return f"Spectograms/{name}_spectrogram.png"
        else:
            return "Spectograms/spectrogram.png"

    def mask_filename(name=None):
        if name:
            return f"Spectograms/{name}_mask.png"
        else:
            return "Spectograms/mask.png"

    def audio_filename(name=None):
        if name:
            return f"Audio/{name}.wav"
        else:
            return "Audio/reconstructed.wav"

    # -------------------------
    # Forward process
    # -------------------------

    sample_rate, audio = utils.load_audio(
        input_file,
        duration=duration
    )

    spectra = stft.calculatr_stft(
        audio,
        frame_size,
        hop_size
    )

    print(f"spectra.shape: {spectra.shape}")

    # -------------------------
    # NMF
    # -------------------------

    W, H, nmf_mag = nmf.calculate_nmf(
        spectra,
        n_components=nmf_components
    )

    # nmf.save_nmf_visualizations(
    #     W,
    #     H,
    #     sample_rate=44100,
    #     n_fft=1024,
    #     output_dir="Spectograms/nmf"
    # )

    # masks = nmf.nmf_component_masks(W, H)
    
    # for mask_index, mask in enumerate(masks):

    #     component_spectra = spectra * mask

    #     audio = stft.calculate_istft(component_spectra)

    #     audio *= 5

    #     utils.save_audio(
    #         audio_filename(f"nmf/component_{mask_index}"),
    #         sample_rate,
    #         audio
    #     )

    # -------------------------
    # Custom separation
    # -------------------------

    # comps = [9, 10, 11, 12, 13, 16, 18, 21, 22, 24, 25, 29]
    comps = []
    comps.append(0)
    comps.append(1)
    comps.append(2)
    comps.append(3)
    comps.append(4)
    comps.append(5)
    comps.append(6)
    comps.append(7)
    comps.append(8)
    # comps.append(9)
    comps.append(10)
    # comps.append(11)
    # comps.append(12)
    # comps.append(13)
    comps.append(14)
    comps.append(15)
    comps.append(16)
    comps.append(17)
    # comps.append(18)
    comps.append(19)
    # comps.append(20)
    # comps.append(21)
    # comps.append(22)
    # comps.append(23)
    # comps.append(24)
    # comps.append(25)
    comps.append(26)
    # comps.append(27)
    comps.append(28)
    # comps.append(29)

    comps = [1, 4, 8, 9]
    comps = [i for i in comps if i < nmf_components]

    print(f"Selected components: {comps}")

    custom_spectra = nmf_sep.get_custom_spectra(
        W, H,
        spectra,
        comps
    )

    utils.save_spectrogram(
        utils.calculate_magnitude(custom_spectra),
        sample_rate,
        hop_size,
        spectogram_filename("custom_non_vocal")
    )

    custom_audio = stft.calculate_istft(
        custom_spectra,
        frame_size,
        hop_size
    )

    utils.save_audio(
        audio_filename("custom_non_vocal"),
        sample_rate,
        custom_audio
    )

    print(f"Saved custom audio to {audio_filename('custom_non_vocal')}")


    comps = [i for i in range(nmf_components) if i not in comps]
    print(f"Selected components: {comps}")

    custom_spectra = nmf_sep.get_custom_spectra(
        W, H,
        spectra,
        comps
    )

    utils.save_spectrogram(
        utils.calculate_magnitude(custom_spectra),
        sample_rate,
        hop_size,
        spectogram_filename("custom_vocal")
    )

    custom_audio = stft.calculate_istft(
        custom_spectra,
        frame_size,
        hop_size
    )

    utils.save_audio(
        audio_filename("custom_vocal"),
        sample_rate,
        custom_audio
    )

    print(f"Saved custom audio to {audio_filename('custom_vocal')}")


    comps = range(nmf_components)
    restored = nmf_sep.get_custom_spectra(
        W, H,
        spectra,
        comps,
        power=2
    )

    # restored = nmf.reconstruct_spectra(
    #     W,
    #     H,
    #     spectra
    # )

    utils.save_spectrogram(
        utils.calculate_magnitude(restored),
        sample_rate,
        hop_size,
        spectogram_filename("restored")
    )

    custom_audio = stft.calculate_istft(
        restored,
        frame_size,
        hop_size
    )

    utils.save_audio(
        audio_filename("restored"),
        sample_rate,
        custom_audio
    )


if __name__ == "__main__":
    main()