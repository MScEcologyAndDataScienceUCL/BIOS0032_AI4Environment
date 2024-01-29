"""Plotting functions for audio data."""

import os
from time import perf_counter

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import xarray as xa
from IPython.display import Audio
from librosa import display
from matplotlib.patches import Rectangle

from bios0032utils.bioacoustics.evaluate_detection import (
    bboxes_from_annotations,
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
        hop_length=hop_length,
        x_axis="time",
        y_axis="linear",
        ax=ax2,
        cmap=cmap,
    )

    plt.tight_layout()
    plt.show()

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
                recording_id: ID of the recording.
                start_time: Start time of the detection.
                end_time: End time of the detection.
                low_freq: Lowest frequency of the detection.
                high_freq: Highest frequency of the detection.

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
        hop_length=hop_length,
        x_axis="time",
        y_axis="linear",
        cmap=cmap,
        ax=ax,
    )

    for _, row in detections.iterrows():
        if str(row["recording_id"]) != os.path.basename(path):
            continue

        rect = Rectangle(
            (row["start_time"], row["low_freq"]),
            row["end_time"] - row["start_time"],
            row["high_freq"] - row["low_freq"],
            linewidth=linewidth,
            edgecolor="r",
            facecolor="none",
        )

        ax.add_patch(rect)  # type: ignore

    plt.show()

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
    iou_threshold=0.5,
    linewidth=1,
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
                recording_id: Name of the recording.
                start_time: Start time of the annotation in seconds.
                end_time: End time of the annotation in seconds.
                low_freq: Minimum frequency of the annotation in Hz.
                high_freq: Maximum frequency of the annotation in Hz.
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
        hop_length=hop_length,
        x_axis="time",
        y_axis="linear",
        cmap=cmap,
        ax=ax,
    )

    annotations = annotations[
        annotations["recording_id"] == os.path.basename(path)
    ]
    predictions = predictions[
        predictions["recording_id"] == os.path.basename(path)
    ]

    true_boxes = bboxes_from_annotations(annotations)
    pred_boxes = bboxes_from_annotations(predictions)
    matches = match_bboxes(true_boxes, pred_boxes, iou_threshold=iou_threshold)

    matched_annotations = set(matches["annotation"])
    matched_predictions = set(matches["prediction"])

    for index, (_, row) in enumerate(annotations.iterrows()):
        if row["recording_id"] != os.path.basename(path):
            continue

        rect = Rectangle(
            (row["start_time"], row["low_freq"]),
            row["end_time"] - row["start_time"],
            row["high_freq"] - row["low_freq"],
            linewidth=linewidth,
            edgecolor="g" if index in matched_annotations else "w",
            facecolor="none",
            linestyle="--",
        )

        ax.add_patch(rect)  # type: ignore

    for index, (_, row) in enumerate(predictions.iterrows()):
        if row["recording_id"] != os.path.basename(path):
            continue

        rect = Rectangle(
            (row["start_time"], row["low_freq"]),
            row["end_time"] - row["start_time"],
            row["high_freq"] - row["low_freq"],
            linewidth=linewidth,
            edgecolor="g" if index in matched_predictions else "r",
            facecolor="none",
        )

        ax.add_patch(rect)  # type: ignore

    plt.show()

    return ax


def plot_spectrogram_with_plotly(
    path: str,
    start_time: float,
    end_time: float,
    low_freq: float,
    high_freq: float,
    context: float = 0.1,
    n_fft: int = 512,
    hop_length: int = 128,
    window: str = "hann",
    eps=0.005,
):
    center = (start_time + end_time) / 2
    offset = center - context / 2

    # compute the spectrogram
    wav, samplerate = librosa.load(
        path,
        sr=None,  # type: ignore
        offset=offset,
        duration=context,
    )

    duration = len(wav) / samplerate

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

    num_freq_bins, num_time_bins = spectrogram.shape
    times = np.linspace(0, duration, num_time_bins)
    freqs = np.linspace(0, samplerate / 2, num_freq_bins)

    xarray = xa.DataArray(
        spectrogram,
        coords=[("freqs", freqs), ("times", times)],
        dims=["freqs", "times"],
    )
    xarray.name = "amplitude"
    xarray.attrs["units"] = "dB"

    fig = px.imshow(xarray, origin="lower")

    fig.add_shape(
        type="rect",
        x0=start_time - offset - 0.002,
        x1=end_time - offset + 0.002,
        y0=low_freq - 8000,
        y1=high_freq + 8000,
        xref="x",
        yref="y",
        line_color="cyan",
    )
    return fig
