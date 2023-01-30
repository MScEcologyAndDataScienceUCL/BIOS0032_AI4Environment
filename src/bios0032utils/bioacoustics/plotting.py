"""Plotting functions for audio data."""
import os
from time import perf_counter

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Audio
from librosa import display
from matplotlib.patches import Rectangle

from bios0032utils.bioacoustics.evaluate_detection import (
    bboxes_from_annotations,
    bboxes_from_tadarida_detections,
    match_bboxes,
)

__all__ = [
    "plot_waveform",
    "plot_waveform_with_spectrogram",
    "plot_spectrogram",
    "WINDOW_OPTIONS",
    "COLORMAPS",
]

WINDOW_OPTIONS = [
    "boxcar",
    "triang",
    "blackman",
    "hamming",
    "hann",
    "bartlett",
    "flattop",
    "parzen",
    "bohman",
]
"""List of available window functions."""

COLORMAPS = [
    "Greys",
    "Purples",
    "Blues",
    "Greens",
    "Oranges",
    "Reds",
    "YlOrBr",
    "YlOrRd",
    "OrRd",
    "PuRd",
    "RdPu",
    "BuPu",
    "GnBu",
    "PuBu",
    "YlGnBu",
    "PuBuGn",
    "BuGn",
    "YlGn",
    "PiYG",
    "PRGn",
    "BrBG",
    "PuOr",
    "RdGy",
    "RdBu",
    "RdYlBu",
    "RdYlGn",
    "Spectral",
    "coolwarm",
    "bwr",
    "seismic",
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
]
"""List of available colormaps."""


def plot_waveform(file: str):
    """Plot the waveform of an audio file."""
    audio, samplerate = librosa.load(file, sr=None)  # type: ignore
    plt.figure(figsize=(10, 3))
    plt.title(file)
    display.waveshow(audio, sr=samplerate)
    plt.show()


def plot_waveform_with_spectrogram(
    file: str,
    hop_length: int = 128,
    n_fft: int = 256,
    window: str = "hann",
    cmap: str = "plasma",
    speed: int = 1,
):
    """Plot the waveform and spectrogram of an audio file.

    Args:
        file: Path to the audio file.
        hop_length: Number of samples between successive frames.
        n_fft: Number of samples per frame.
        window: Window function to use.
        cmap: Colormap to use.
        slowdown: Slow down the audio by this factor.

    Returns:
        Audio object to play the audio.

    """
    wav, samplerate = librosa.load(file, sr=None)  # type: ignore

    plt.figure(figsize=(10, 6))

    ax1 = plt.subplot2grid((3, 1), (0, 0))
    ax2 = plt.subplot2grid((3, 1), (1, 0), rowspan=2, sharex=ax1)

    display.waveshow(wav, sr=samplerate, ax=ax1)

    time = perf_counter()
    spectrogram = librosa.amplitude_to_db(
        np.abs(
            librosa.stft(
                wav,
                hop_length=hop_length,
                n_fft=n_fft,
                window=window,
            )
        ),
        ref=np.max,  # type: ignore
    )
    time = perf_counter() - time

    print(f"Time to compute spectrogram: {time:.4f} seconds")
    print(f"Size of spectrogram: {spectrogram.shape}")

    display.specshow(
        spectrogram,
        sr=samplerate,
        n_fft=n_fft,
        hop_length=hop_length,
        x_axis="time",
        y_axis="linear",
        ax=ax2,
        cmap=cmap,
    )

    plt.tight_layout()

    return Audio(data=wav, rate=int(samplerate * speed))


def plot_spectrogram(
    path: str,
    hop_length: int = 128,
    n_fft: int = 256,
    window: str = "hann",
    cmap: str = "plasma",
    figsize=(10, 6),
):
    """Plot the spectrogram of an audio file.

    Args:
        path: Path to the audio file.
        hop_length: Number of samples between successive frames.
        n_fft: Number of samples per frame.
        window: Window function to use.
        cmap: Colormap to use.

    """
    wav, samplerate = librosa.load(path, sr=None)  # type: ignore

    plt.figure(figsize=figsize)

    spectrogram = librosa.amplitude_to_db(
        np.abs(
            librosa.stft(
                wav,
                hop_length=hop_length,
                n_fft=n_fft,
                window=window,
            )
        ),
        ref=np.max,  # type: ignore
    )

    display.specshow(
        spectrogram,
        sr=samplerate,
        n_fft=n_fft,
        hop_length=hop_length,
        x_axis="time",
        y_axis="linear",
        cmap=cmap,
    )


def plot_spectrogram_and_detection(
    path: str,
    detections: pd.DataFrame,
    ax=None,
    figsize=(14, 4),
    cmap="magma",
    n_fft=512,
    hop_length=128,
    linewidth=2,
):
    """Plot the spectrogram of an audio file and detections.

    Draw a red rectangle for each detection.

    Args:
        file: Path to the audio file.
        detections: Dataframe with detections.
            The dataframe must have the following columns:
                StTime: Start time of the detection in milliseconds.
                Dur: Duration of the detection in milliseconds.
                Fmin: Minimum frequency of the detection in kHz.
                BW: Bandwidth of the detection in kHz.

    """
    wav, samplerate = librosa.load(path, sr=None)  # type: ignore

    spectrogram = librosa.amplitude_to_db(
        np.abs(
            librosa.stft(
                wav,
                hop_length=hop_length,
                n_fft=n_fft,
                window="hann",
            )
        ),
        ref=np.max,  # type: ignore
    )

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    display.specshow(
        spectrogram,
        sr=samplerate,
        n_fft=n_fft,
        hop_length=hop_length,
        x_axis="time",
        y_axis="linear",
        cmap=cmap,
        ax=ax,
    )

    for _, row in detections.iterrows():
        if str(row["wav"]) != path:
            continue

        rect = Rectangle(
            (row["StTime"] / 1000, row["Fmin"] * 1000),
            row["Dur"] / 1000,
            row["BW"] * 1000,
            linewidth=linewidth,
            edgecolor="r",
            facecolor="none",
        )

        ax.add_patch(rect)  # type: ignore

    return ax


def plot_spectrogram_and_ground_truth(
    path: str,
    ground_truth: pd.DataFrame,
    ax=None,
    figsize=(14, 4),
    cmap="magma",
    n_fft=512,
    hop_length=128,
    linewidth=2,
):
    """Plot the spectrogram of an audio file and ground truth.

    Draw a red rectangle for each ground truth annotation.

    Args:
        file: Path to the audio file.
        ground_truth: Dataframe with ground truth annotations.
            The dataframe must have the following columns:
                recording_id: Name of the recording.
                start_time: Start time of the annotation in seconds.
                end_time: End time of the annotation in seconds.
                low_freq: Minimum frequency of the annotation in Hz.
                high_freq: Maximum frequency of the annotation in Hz.
    """
    wav, samplerate = librosa.load(path, sr=None)  # type: ignore

    spectrogram = librosa.amplitude_to_db(
        np.abs(
            librosa.stft(
                wav,
                hop_length=hop_length,
                n_fft=n_fft,
                window="hann",
            )
        ),
        ref=np.max,  # type: ignore
    )

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    display.specshow(
        spectrogram,
        sr=samplerate,
        n_fft=n_fft,
        hop_length=hop_length,
        x_axis="time",
        y_axis="linear",
        cmap=cmap,
        ax=ax,
    )

    for _, row in ground_truth.iterrows():
        if row["recording_id"] != os.path.basename(path):
            continue

        rect = Rectangle(
            (row["start_time"], row["low_freq"]),
            row["end_time"] - row["start_time"],
            row["high_freq"] - row["low_freq"],
            linewidth=linewidth,
            edgecolor="r",
            facecolor="none",
        )

        ax.add_patch(rect)

    return ax


def plot_spectrogram_with_predictions_and_annotations(
    path: str,
    predictions: pd.DataFrame,
    annotations: pd.DataFrame,
    ax=None,
    figsize=(14, 4),
    cmap="magma",
    n_fft=512,
    hop_length=128,
    threshold=0.5,
):
    """Plot the spectrogram of an audio file and predictions and annotations.

    The function will compute the matches between predictions and annotations
    and draw a red rectangle for each false positive detection. A green
    rectangle for each true positive detection. And, a white rectangle for each
    missed annotation. All annotations are drawn with dashed lines.

    Args:
        file: Path to the audio file.
        predictions: Dataframe with predictions.
            The dataframe must have the following columns:
                Filename: Name of the recording.
                StTime: Start time of the prediction in milliseconds.
                Dur: Duration of the prediction in milliseconds.
                Fmin: Minimum frequency of the prediction in kHz.
                BW: Bandwidth of the prediction in kHz.
        annotations: Dataframe with ground truth annotations.
            The dataframe must have the following columns:
                recording_id: Name of the recording.
                start_time: Start time of the annotation in seconds.
                end_time: End time of the annotation in seconds.
                low_freq: Minimum frequency of the annotation in Hz.
                high_freq: Maximum frequency of the annotation in Hz.
        matches: Dataframe with matches between predictions and annotations.
            The dataframe must have the following columns:
                prediction_id: ID of the prediction.
                annotation_id: ID of the annotation.
                score: Match score.
    """

    wav, samplerate = librosa.load(path, sr=None)  # type: ignore

    spectrogram = librosa.amplitude_to_db(
        np.abs(
            librosa.stft(
                wav,
                hop_length=hop_length,
                n_fft=n_fft,
                window="hann",
            )
        ),
        ref=np.max,  # type: ignore
    )

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    display.specshow(
        spectrogram,
        sr=samplerate,
        n_fft=n_fft,
        hop_length=hop_length,
        x_axis="time",
        y_axis="linear",
        cmap=cmap,
        ax=ax,
    )

    annotations = annotations[annotations["recording_id"] == os.path.basename(path)]
    predictions = predictions[predictions["Filename"] == os.path.basename(path)]

    true_boxes = bboxes_from_annotations(annotations)
    pred_boxes = bboxes_from_tadarida_detections(predictions)
    matches = match_bboxes(true_boxes, pred_boxes, threshold=threshold)

    matched_annotations = set(matches["annotation"])
    matched_predictions = set(matches["prediction"])

    for index, (_, row) in enumerate(annotations.iterrows()):
        if row["recording_id"] != os.path.basename(path):
            continue

        rect = Rectangle(
            (row["start_time"], row["low_freq"]),
            row["end_time"] - row["start_time"],
            row["high_freq"] - row["low_freq"],
            linewidth=1,
            edgecolor="g" if index in matched_annotations else "w",
            facecolor="none",
            linestyle="--",
        )

        ax.add_patch(rect)  # type: ignore


    for index, (_, row) in enumerate(predictions.iterrows()):
        if row["Filename"] != os.path.basename(path):
            continue

        rect = Rectangle(
            (row["StTime"] / 1000, row["Fmin"] * 1000),
            row["Dur"] / 1000,
            row["BW"] * 1000,
            linewidth=2,
            edgecolor="r" if index not in matched_predictions else "g",
            facecolor="none",
        )

        ax.add_patch(rect) # type: ignore

    return ax
