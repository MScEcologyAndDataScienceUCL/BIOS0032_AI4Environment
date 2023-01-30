import pandas as pd
from pytadarida import run_tadarida


def run_tadarida_detection(files):
    detection, _ = run_tadarida(files)
    return pd.DataFrame(
        {
            "recording_id": detection.Filename,
            "start_time": detection.StTime / 1000,
            "end_time": (detection.StTime + detection.Dur) / 1000,
            "low_freq": detection.Fmin * 1000,
            "high_freq": (detection.Fmin + detection.BW) * 1000,
        }
    )
