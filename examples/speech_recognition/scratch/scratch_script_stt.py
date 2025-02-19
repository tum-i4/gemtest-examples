###############################################################################################
#     This script is just an experimental prototype, and not a core part of the examples.     #
###############################################################################################
import os
import warnings
from glob import glob
from pathlib import Path

import soundfile  # type: ignore
import torch
from audiomentations import Compose, AddGaussianNoise  # type: ignore
from audiomentations import PitchShift, AddBackgroundNoise  # type: ignore

warnings.filterwarnings("ignore")


def write_audio(audio, sr, file_name, dir_path):
    if not dir_path:
        return
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    soundfile.write(os.path.join(dir_path, file_name), audio, sr)


def transform(source_audio, source_sample_rate, n_out_puts=5,
              out_dir="", verbose=False):
    if isinstance(source_audio, torch.Tensor):
        source_audio = source_audio.numpy()
    source_audio = source_audio.reshape((-1,))  # only for mono audio
    follow_up_audio_list = []
    for idx in range(n_out_puts):
        aug_audio = augment(source_audio, source_sample_rate)
        follow_up_audio_list.append(torch.from_numpy(aug_audio).reshape(1, -1))
        write_audio(aug_audio, source_sample_rate, f"augmented_audio_{idx + 1}.wav", out_dir)
    if verbose:
        if out_dir:
            print(f"{n_out_puts} augmented audio files are written to {out_dir}!!!")
        else:
            print("No files written as out_dir is None!!!")
    return follow_up_audio_list


if __name__ == "__main__":
    device = torch.device('cpu')  # gpu also works, but our models are fast enough for CPU
    model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                           model='silero_stt',
                                           language='en',  # also available 'de', 'es'
                                           device=device)
    (read_batch, split_into_batches,
     read_audio, prepare_model_input) = utils  # see function signature for details

    # download a single file in any format compatible with TorchAudio
    torch.hub.download_url_to_file(
        'https://opus-codec.org/static/examples/samples/speech_orig.wav',
        dst='speech_orig.wav', progress=True)
    test_files = glob('speech_orig.wav')
    batches = split_into_batches(test_files, batch_size=10)

    input_ = prepare_model_input(read_batch(batches[0]),
                                 device=device)

    augment = Compose([
        AddGaussianNoise(min_amplitude=0.0001, max_amplitude=0.001, p=0.3),
        AddBackgroundNoise(sounds_path=["../background_noises"], p=0.7),
        PitchShift(min_semitones=-2, max_semitones=+2, p=0.5),
    ], shuffle=True)

    followup_inputs = transform(input_, 16000)

    output = model(input_).squeeze(0)  # output shape: 136 x 999
    followup_outputs = [model(inp).squeeze(0) for inp in
                        followup_inputs]  # op shape : 136 x 999

    # look at the recognized texts for original and followup inputs
    print("\n\n")
    for i, example in enumerate([output] + followup_outputs):
        if i == 0:
            print(f"Original Speech2Text: '{decoder(example.cpu())}'\n")
        else:
            print(f"Augmented Speech2Text_{i}: '{decoder(example.cpu())}'\n")
