from pathlib import Path
from typing import Union, List

import librosa  # type: ignore
import numpy as np
import torch
from audiomentations import Compose, AddGaussianNoise, PitchShift, TimeStretch  # type: ignore

import gemtest as gmt
from examples.speech_recognition.audio_visualizer import AudioVisualizer  # type: ignore
from .models.aec_inference import AudioTagging, labels  # type: ignore

# region model
# creating this model object outside the test not to load it again and again for each test
DEVICE = 'cpu'
at = AudioTagging(device=DEVICE)
# endregion

# region data
src_audios = list(
    librosa.core.load(
        str(
            Path("") /
            "examples" /
            "acoustic_scene_classifier" /
            "acoustic_event_samples" /
            f"test_aec_audio_{i}.wav"
        ),
        sr=32000,
        mono=True
    )[0] for i in range(0, 10)  # [0] Indicates we are loading only the audio data (1st output)
)


# endregion

# region auxiliary function
def get_top_k_labels(clip_wise_output: np.ndarray, top_k: int = 5) -> List[str]:
    """
    Returns a list of top k predicted labels

    Args
    ----
    clip_wise_output: np.ndarray
        Numpy array of shape (number of classes, ) containing soft-max probabilities of all
        possible class labels.
    top_k: int
        number upto which the sorted label probabilities needs to be considered for comparison

    Returns
    -------
    list of top k labels: List[str]
    """
    sorted_indexes = np.argsort(clip_wise_output)[::-1]
    return [np.array(labels)[sorted_indexes[k]] for k in range(top_k)]


# audio visualizer for aec
aec_audio_visualizer = AudioVisualizer(
    sampling_rate=32000,
    base_dir=str(Path("") / "assets" / "aec")
)


# endregion

# region noise functions
def add_gaussian_noise(
        source_audio: Union[np.ndarray, torch.Tensor],
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
        return torch.from_numpy(transform(source_audio, 32000))
    return torch.from_numpy(transform(source_audio.numpy(), 32000))  # type: ignore


def add_background_noise(
        source_audio: Union[np.ndarray, torch.Tensor],
        p: float
) -> torch.Tensor:
    """
    This transformation adds a random background noise from the sounds_path folder
    to the source_audio with probability p and returns that transformed audio.

    params:
        source_audio: numpy ndarray of shape (<number of samples>,): the input audio
        p: float: probability of applying the transformation

    returns:
        torch tensor of shape (<number of samples>,) (same shape of input)
    """
    transform = TimeStretch(p=p)
    if not torch.is_tensor(source_audio):
        return torch.from_numpy(transform(source_audio, 32000))
    return torch.from_numpy(transform(source_audio.numpy(), 32000))  # type: ignore


def add_pitch_noise(
        source_audio: Union[np.ndarray, torch.Tensor],
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
        return torch.from_numpy(transform(source_audio, 32000))
    return torch.from_numpy(transform(source_audio.numpy(), 32000))  # type: ignore


# endregion

# region test_names
with_gaussian_noise = gmt.create_metamorphic_relation(
    name='with_gaussian_noise',
    data=src_audios)

with_background_noise = gmt.create_metamorphic_relation(
    name='with_background_noise',
    data=src_audios)

with_pitch_noise = gmt.create_metamorphic_relation(
    name='with_pitch_noise',
    data=src_audios)

with_gaussian_background_noise = gmt.create_metamorphic_relation(
    name='with_gaussian_background_noise',
    data=src_audios)

with_background_pitch_noise = gmt.create_metamorphic_relation(
    name='with_background_pitch_noise',
    data=src_audios)

with_combined_noise = gmt.create_metamorphic_relation(
    name='with_combined_noise',
    data=src_audios)


# endregion

# region transformations
@gmt.transformation(with_gaussian_noise)
@gmt.fixed('min_amplitude', 0.001)
@gmt.fixed('max_amplitude', 0.01)
@gmt.fixed('p', 1.)
def transform_gauss(source_audio: Union[np.ndarray, torch.Tensor], min_amplitude: float,
                    max_amplitude: float, p: float) -> torch.Tensor:
    return add_gaussian_noise(source_audio, min_amplitude, max_amplitude, p)


@gmt.transformation(with_background_noise)
@gmt.fixed('p', 1.)
def transform_background(source_audio: Union[np.ndarray, torch.Tensor], p: float) \
        -> torch.Tensor:
    return add_background_noise(source_audio, p)


@gmt.transformation(with_pitch_noise)
@gmt.fixed('min_semitones', -3)
@gmt.fixed('max_semitones', 3)
@gmt.fixed('p', 1.)
def transform_pitch(source_audio: Union[np.ndarray, torch.Tensor], min_semitones: int,
                    max_semitones: int, p: float) -> torch.Tensor:
    return add_pitch_noise(source_audio, min_semitones, max_semitones, p)


@gmt.transformation(with_gaussian_background_noise)
@gmt.fixed('min_amplitude', 0.001)
@gmt.fixed('max_amplitude', 0.01)
@gmt.fixed('p', 1.)
def transform_gauss(source_audio: Union[np.ndarray, torch.Tensor], min_amplitude: float,
                    max_amplitude: float, p: float) -> torch.Tensor:
    gaussian_audio = add_gaussian_noise(source_audio, min_amplitude, max_amplitude, p)
    return add_background_noise(gaussian_audio, p)


@gmt.transformation(with_background_pitch_noise)
@gmt.fixed('min_semitones', -3)
@gmt.fixed('max_semitones', 3)
@gmt.fixed('p', 1.)
def transform_pitch(source_audio: Union[np.ndarray, torch.Tensor], min_semitones: int,
                    max_semitones: int, p: float) -> torch.Tensor:
    background_audio = add_background_noise(source_audio, p)
    return add_pitch_noise(background_audio, min_semitones, max_semitones, p)


@gmt.transformation(with_combined_noise)
def composite_transformation(
        source_audio: Union[np.ndarray, torch.Tensor],
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
            TimeStretch(p=0.5),
            PitchShift(min_semitones=-3, max_semitones=+3, p=0.5),
        ], shuffle=True)
    if not torch.is_tensor(source_audio):
        return torch.from_numpy(transform(source_audio, 32000))
    return torch.from_numpy(transform(source_audio.numpy(), 32000))  # type: ignore


# endregion

# region custom_relation
# relation is registered on all MRs if none are specified
@gmt.general_relation()
def aec_top_k_compare(mtc: gmt.MetamorphicTestCase) -> bool:
    """
    This is a custom metamorphic comparison relation designed specifically for acoustic
    event classification algorithms. There can be multiple class label predictions for a given
    audio. So, instead of comparing the class labels one by one, we consider top k (say k=5)
    predictions for both source input and followup inputs and try to see if there is any
    intersection between them. Test passes if there is intersection, fails otherwise.

    params:
        f_x: List[str]: list of top k labels predicted for source input
        f_xt: List[str]: list of top k labels predicted for followup input
        x: Union[np.ndarray, torch.Tensor]: input audio of shape
            (<number of samples>,)
        x_t: torch.Tensor: Transformed input of shape (<number of samples>,)
            (same shape of input)

    returns:
        bool: True refers to a passing test
    """
    gmt.logger.info("Source Input Predictions: %s", str(mtc.source_outputs))
    gmt.logger.info("Followup Input Predictions: %s", str(mtc.followup_outputs))

    return bool(set(mtc.source_outputs) & set(mtc.followup_outputs))


# endregion

# region system under test
@gmt.system_under_test(visualize_input=aec_audio_visualizer)
def test_acoustic_event_classification(audio):  # acoustic event classification (aec) test
    audio = audio[None, :]
    (clip_wise_output, _) = at.inference(audio)
    predictions = get_top_k_labels(np.squeeze(clip_wise_output), top_k=5)
    return predictions
# endregion
