"""Classification of bat calls utils."""

import wave
from typing import Optional, Tuple

import numpy as np
import wavio

TADARIDA_FEATURES = [
    "Dur",
    "Fmin",
    "BW",
    "PosMP",
    "PrevMP1",
    "PrevMP2",
    "NextMP1",
    "NextMP2",
    "Amp1",
    "Amp2",
    "Amp3",
    "Amp4",
    "NoisePrev",
    "NoiseNext",
    "NoiseDown",
    "NoiseUp",
    "CVAmp",
    "CO2_FPkD",
    "CO2_TPk",
    "CM_Slope",
    "CS_Slope",
    "CN_Slope",
    "CO_Slope",
    "CO2_Slope",
    "CO2_ISlope",
    "CM_THCF",
    "CS_THCF",
    "CN_THCF",
    "CO_THCF",
    "CO2_THCF",
    "CM_FIF",
    "CS_FIF",
    "CN_FIF",
    "CM_UpSl",
    "CS_UpSl",
    "CN_UpSl",
    "CO_UpSl",
    "CO2_UpSl",
    "CM_LoSl",
    "CS_LoSl",
    "CN_LoSl",
    "CO_LoSl",
    "CO2_LoSl",
    "CM_StSl",
    "CS_StSl",
    "CN_StSl",
    "CO_StSl",
    "CO2_StSl",
    "CM_EnSl",
    "CS_EnSl",
    "CN_EnSl",
    "CO_EnSl",
    "CO2_EnSl",
    "CS_FPSl",
    "CN_FPSl",
    "CO_FPSl",
    "CM_FISl",
    "CO2_FISl",
    "CM_5dBBW",
    "CM_5dBDur",
    "CO2_5dBBW",
    "CO2_5dBDur",
    "Hup_RFMP",
    "Hup_AmpDif",
    "Hlo_PosEn",
    "Hlo_AmpDif",
    "Ramp_2_1",
    "Ramp_3_1",
    "Ramp_3_2",
    "Ramp_1_2",
    "Ramp_2_3",
    "RAN_2_1",
    "RAN_3_1",
    "RAN_3_2",
    "RAN_1_2",
    "RAN_4_3",
    "RAN_2_3",
    "HetX",
    "Dbl8",
    "Stab",
    "HeiET",
    "HeiEM",
    "HeiRT",
    "HeiRM",
    "HeiEMT",
    "HeiRTT",
    "HeiRMT",
    "Int25",
    "Int75",
    "RInt1",
    "SmIntDev",
    "LgIntDev",
    "VarInt",
    "VarSmInt",
    "VarLgInt",
    "RIntDev1",
    "EnStabSm",
    "EnStabLg",
    "HetYr",
    "HetCMC",
    "HetCMD",
    "HetCTC",
    "HetCTD",
    "HetCMfP",
    "HetCTfP",
    "HetPicsMALD",
    "HetPicsMABD",
    "HetPicsMRLBD",
    "HetPicsTABD",
    "HetPicsTRLBD",
    "VLDPPicsM",
    "VBDPPicsM",
    "VLDPPicsT",
    "VBDPPicsT",
    "CM_SDCR",
    "CS_SDCR",
    "CN_SDCR",
    "CO_SDCR",
    "CO2_SDCR",
    "CM_SDCRXY",
    "CS_SDCRXY",
    "CM_SDCL",
    "CM_SDCLOP",
    "CM_SDCLROP",
    "CM_SDCLRWB",
    "CM_SDCLRXYOPWB",
    "CM_SDCLR_DNP",
    "CS_SDCLOP",
    "CS_SDCLROP",
    "CS_SDCLRYOP",
    "CS_SDCLWB",
    "CS_SDCLR_DNP",
    "CS_SDCLRY_DNP",
    "CM_ELBPOS",
    "CS_ELBPOS",
    "CM_ELBSB",
    "CS_ELBSB",
    "CM_ELB2POS",
    "CS_ELB2POS",
    "CM_ELB2SB",
    "CS_ELB2SB",
    "CM_RAFE",
    "CM_RAFP3",
    "CM_SBMP",
    "CM_SAMP",
    "CM_SBAR",
    "RAHE4",
]
"""List of Tadarida features."""


WINDOW_DURATION = 15360 / 441000
SAMPLERATE = 441000


def load_bat_call_audio_data(
    path: str,
    start_time: float,
    end_time: float,
    duration: float = WINDOW_DURATION,
) -> np.ndarray:
    """Load bat call audio data.

    This function will load a chunk of audio from the given
    audio file of the given duration but centered on the given start and end times.

    Args:
        path (str): Path to the audio file.
        start_time (float): Start time of the bat call.
        end_time (float): End time of the bat call.
        duration (float, optional): Duration of the extracted audio chunk. Defaults to WINDOW_DURATION.
    """
    center = (start_time + end_time) / 2
    start = center - (duration / 2)
    end = start + duration

    wav, samplerate = load_audio(
        path,
        start_time=start,
        end_time=end,
    )

    assert samplerate == SAMPLERATE

    return wav.squeeze()


def load_audio(
    path: str,
    start_time: float = 0,
    end_time: Optional[float] = None,
) -> Tuple[np.ndarray, int]:
    """Load audio from wav file

    Can specify load starting time and ending time, which is particularly
    helpful to quickly retrieve a small audio clip without loading the full
    file.

    If `start_time` is not provided audio will be loaded from the start
    of the recording. If `end_time` is not provided loading will stop at
    the end of the recording.

    If `start_time` or `end_time` fall outside the recording's duration the
    returned wav will be padded with 0's.

    Parameters
    ----------
    path: str
        Path to audio file in filesystem
    start_time: float
        Time from which to start loading audio, in seconds. Time is
        relative to the start of the recording and computed based on the
        recording's samplerate. If the recording is time expanded, adjust the
        `start_time` accordingly. Defaults to 0.
    end_time: Optional[float]
        Time at which to end loading audio, in seconds. Time is
        relative to the start of the recording and computed based on the
        recording's samplerate. If the recording is time expanded, adjust the
        `end_time` accordingly.

    Returns
    -------
    wav: numpy.ndarray
        2D-Array with the loaded audio data. The array is two dimensional
        with shape [nframes, channels].
    sr: int
        Samplerate of the audio file

    """
    with wave.open(path, "rb") as wav_file:
        # Read wav file parameters
        params = wav_file.getparams()

        samplerate = params.framerate
        length = params.nframes
        channels = params.nchannels

        start = int(np.floor(start_time * samplerate))

        if end_time is None:
            end = length
        else:
            end = int(np.floor(end_time * samplerate))

        if (start >= length) or (end <= 0):
            return np.zeros([end - start, channels]), samplerate

        if start > 0:
            wav_file.setpos(start)  # type: ignore

        extra_start = min(start, 0)
        start = max(start, 0)

        extra_end = max(end, length) - length
        end = min(end, length)

        data = wav_file.readframes(end - start)  # type: ignore

        # Turn into numpy array
        wav: np.ndarray = wavio._wav2array(
            channels,
            params.sampwidth,
            data,
        )

        # Normalize to [-1, 1]
        wav = wav / np.iinfo(wav.dtype).max

        # Pad with 0s if start_time of end_time extends over the recording
        # interval
        if (extra_start < 0) or (extra_end > 0):
            wav = np.pad(
                wav,
                [  # type: ignore
                    [-extra_start, extra_end],
                    [0, 0],
                ],
            )

        return wav, samplerate
