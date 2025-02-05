from pathlib import Path
from typing import Union, List

import numpy
import torch
from audiomentations import Compose, AddGaussianNoise  # type: ignore
from audiomentations import PitchShift, AddBackgroundNoise  # type: ignore
from jiwer import wer, mer, wil  # type: ignore

import gemtest as gmt
from examples.speech_recognition.audio_visualizer import AudioVisualizer  # type: ignore
from examples.speech_recognition.models.speech_to_text import SpeechToText  # type: ignore
from examples.speech_recognition.utils.stt_utils import stt_read_audio  # type: ignore
from gemtest.logger import logger


# define functions used for transformations.
# mt-framework 0.3. does not support chaining of transformations - this is a workaround
def add_gaussian_noise(
        source_audio: Union[numpy.ndarray, torch.Tensor],
        min_amplitude: float,
        max_amplitude: float,
        p: float
) -> torch.Tensor:
    """
    This transformation adds a random Gaussian noise (within min_amplitude and max_amplitude)
    to the source_audio with probability p and returns that transformed audio.

    params:
        source_audio: numpy ndarray of shape (<number of samples>,): the input audio
        min_amplitude: float: minimum amplitude of the Gaussian noise
        max_amplitude: float: maximum amplitude of the Gaussian noise
        p: float: probability of applying the transformation

    returns:
        torch tensor of shape (<number of samples>,) (same shape of input)
    """
    transform = AddGaussianNoise(min_amplitude=min_amplitude, max_amplitude=max_amplitude, p=p)
    if not torch.is_tensor(source_audio):
        return torch.from_numpy(transform(source_audio, 16000))
    return torch.from_numpy(transform(source_audio.numpy(), 16000))  # type: ignore


def add_background_noise(
        source_audio: Union[numpy.ndarray, torch.Tensor],
        sounds_path: Union[List[Path], List[str], Path, str],
        p: float
) -> torch.Tensor:
    """
    This transformation adds a random background noise from the sounds_path folder
    to the source_audio with probability p and returns that transformed audio.

    params:
        source_audio: numpy ndarray of shape (<number of samples>,): the input audio
        sounds_path: A path or list of paths to audio file(s) and/or folder(s) with
            audio files. Can be str or Path instance(s). The audio files given here are
            supposed to be background noises.
        p: float: probability of applying the transformation

    returns:
        torch tensor of shape (<number of samples>,) (same shape of input)
    """
    transform = AddBackgroundNoise(sounds_path=sounds_path, p=p)
    if not torch.is_tensor(source_audio):
        return torch.from_numpy(transform(source_audio, 16000))
    return torch.from_numpy(transform(source_audio.numpy(), 16000))


def alter_pitch(
        source_audio: Union[numpy.ndarray, torch.Tensor],
        min_semitones: int,
        max_semitones: int,
        p: float
) -> torch.Tensor:
    """
    This transformation randomly alters the pitch of the source_audio within min_semitones and
    max_semitones with probability p and returns the transformed audio

    params:
        source_audio: numpy ndarray of shape (<number of samples>,): the input audio
        min_semitones: int: lower limit of the pitch alteration. A -ve value implies pitch
            should be lowered from the pitch of the source_audio
        max_semitones: int: upper limit of the pitch alteration
         p: float: probability of applying the transformation

    returns:
        torch tensor of shape (<number of samples>,) (same shape of input)
    """
    transform = PitchShift(min_semitones=min_semitones, max_semitones=max_semitones, p=p)
    if not torch.is_tensor(source_audio):
        return torch.from_numpy(transform(source_audio, 16000))
    return torch.from_numpy(transform(source_audio.numpy(), 16000))  # type: ignore


# region model
# creating this model object outside the test not to load it again and again for each test
stt = SpeechToText()
# endregion


# region data_list
# src_audios is the list of audios with which the test need to be performed
src_audios = list(
    stt_read_audio(
        str(
            Path("") / "examples" / "speech_recognition" / "speech_samples" /
            f"test_audio_{i}.wav"
        )
    ) for i in range(0, 5)
)
# endregion


# region test_names
# register the metamorphic testcases for speech recognition
with_gaussian_noise = gmt.create_metamorphic_relation('with_gaussian_noise', src_audios)
with_background_noise = gmt.create_metamorphic_relation('with_background_noise', src_audios)
with_altered_pitch = gmt.create_metamorphic_relation('with_altered_pitch', src_audios)
with_combined_effect = gmt.create_metamorphic_relation('with_combined_effect', src_audios)
with_chained_transform_a = gmt.create_metamorphic_relation('with_chained_transform_a',
                                                           src_audios)
with_chained_transform_b = gmt.create_metamorphic_relation('with_chained_transform_b',
                                                           src_audios)


# endregion

# a = gaus + background
# b = background + altered pitch

# region custom_transformations

# define and register the audio transformations
# transformation to add Gaussian noise:
@gmt.transformation(with_gaussian_noise)
@gmt.fixed('min_amplitude', 0.0001)
@gmt.fixed('max_amplitude', 0.001)
@gmt.fixed('p', 1.)
def gaussian_noise_transformation(
        source_audio: Union[numpy.ndarray, torch.Tensor],
        min_amplitude: float,
        max_amplitude: float,
        p: float
) -> torch.Tensor:
    return add_gaussian_noise(source_audio, min_amplitude, max_amplitude, p)


# transformation to add background noise
@gmt.transformation(with_background_noise)
@gmt.fixed('sounds_path', 'examples/speech_recognition/background_noises')
@gmt.fixed('p', 1.)
def background_noise_transformation(
        source_audio: Union[numpy.ndarray, torch.Tensor],
        sounds_path: Union[List[Path], List[str], Path, str],
        p: float
) -> torch.Tensor:
    return add_background_noise(source_audio, sounds_path, p)


# transformation to alter pitch
@gmt.transformation(with_altered_pitch)
@gmt.fixed('min_semitones', -2)
@gmt.fixed('max_semitones', 2)
@gmt.fixed('p', 1.)
def alter_pitch(
        source_audio: Union[numpy.ndarray, torch.Tensor],
        min_semitones: int,
        max_semitones: int,
        p: float
) -> torch.Tensor:
    return alter_pitch(source_audio, min_semitones, max_semitones, p)


@gmt.transformation(with_chained_transform_a)
@gmt.fixed('min_amplitude', 0.0001)
@gmt.fixed('max_amplitude', 0.001)
@gmt.fixed('sounds_path', 'examples/speech_recognition/background_noises')
@gmt.fixed('p', 1.)
def combined_gaussian_background(
        source_audio: Union[numpy.ndarray, torch.Tensor],
        min_amplitude: float,
        max_amplitude: float,
        sounds_path: str,
        p: float
) -> torch.Tensor:
    gaussian = add_gaussian_noise(source_audio, min_amplitude, max_amplitude, p)
    return add_background_noise(gaussian, sounds_path, p)


@gmt.transformation(with_chained_transform_b)
@gmt.fixed('sounds_path', 'examples/speech_recognition/background_noises')
@gmt.fixed('min_semitones', -2)
@gmt.fixed('max_semitones', 2)
@gmt.fixed('p', 1.)
def combined_background_pitch(
        source_audio: Union[numpy.ndarray, torch.Tensor],
        sounds_path: str,
        min_semitones: int,
        max_semitones: int,
        p: float
) -> torch.Tensor:
    background = add_background_noise(source_audio, sounds_path, p)
    return alter_pitch(background, min_semitones, max_semitones, p)


# combined transformation
@gmt.transformation(with_combined_effect)
def composite_transformation(
        source_audio: Union[numpy.ndarray, torch.Tensor]
) -> torch.Tensor:
    """
    This composite transformation probabilistically combines the above three transformations
    in random order. Each of the three basic transforms (AddGaussianNoise, AddBackgroundNoise,
    PitchShift) has 50% percent chance of being used in this transformation. Chosen transforms
    are then applied in a randomly shuffled order.

    params:
        source_audio: Union[numpy.ndarray, torch.Tensor]: input audio of shape
                    (<number of samples>,)
    returns:
        torch tensor of shape (<number of samples>,) (same shape of input)
    """
    transform = Compose(
        [
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=0.5),
            AddBackgroundNoise(
                sounds_path=["examples/speech_recognition/background_noises"],
                p=0.5
            ),
            PitchShift(min_semitones=-3, max_semitones=+3, p=0.5),
        ], shuffle=True)
    if not torch.is_tensor(source_audio):
        return torch.from_numpy(transform(source_audio, 16000))
    return torch.from_numpy(transform(source_audio.numpy(), 16000))  # type: ignore


# endregion


# region custom_relations

@gmt.relation()
def stt_soft_compare(source_output: str, followup_output: str) -> bool:
    """
    This is a custom metamorphic comparison relation designed specifically for speech2text
    algorithms. Direct string comparison for recognized texts from source and followup cases
    can be too restrictive and too harsh on the speech recognition algorithm under test.

    So, we use standard latent for speech recognition algorithms, namely:
    Word Error Rate (WER), Matching Error Rate (MER) and Word Information Loss (WIL) and
    consider our test to be passing if those latent are below certain predefined threshold.

    params:
        f_x: str: recognize text from source test case
        f_xt: str: recognized text from follow-up test case

    returns:
        bool: True refers to a passing test
    """
    wer_val = wer(source_output, followup_output)
    mer_val = mer(source_output, followup_output)
    wil_val = wil(source_output, followup_output)
    # logger.info(f"WER:{wer_val}, MER:{mer_val}, WIL:{wil_val}")  # linter breaks
    logger.info("WER: %0.3f, MER: %0.3f, WIL: %0.3f", wer_val, mer_val, wil_val)
    return wer_val <= 0.3 and mer_val <= 0.3 and wil_val <= 0.5  # empirically chosen threshold


# endregion


# region visualizer

stt_audio_visualizer = AudioVisualizer(
    sampling_rate=16000,
    base_dir=str(Path("") / "assets" / "stt")
)


# endregion


# region system under test
# Mark this function as the system under test
@gmt.system_under_test(visualize_input=stt_audio_visualizer)
def test_stt(audio):
    return stt.recognize(audio)
# endregion
