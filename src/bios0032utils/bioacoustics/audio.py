from io import BytesIO

import numpy as np
from ipywebrtc import AudioRecorder, CameraStream
from pydub import AudioSegment


def create_audio_recorder() -> AudioRecorder:
    """Create an audio recorder widget."""
    return AudioRecorder(
        stream=CameraStream(constraints={"audio": True, "video": False})
    )


def extract_wav_from_recorder(recorder: AudioRecorder) -> np.ndarray:
    """Extract a wav audio segment from an audio recorder widget."""
    audio = BytesIO()
    audio.write(recorder.audio.value)
    audio.seek(0)
    return AudioSegment.from_file(audio, format="webm").get_array_of_samples()
