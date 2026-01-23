import numpy as np
import sounddevice as sd


def calibrate_mic(cal_spl_db=114, channel=0, duration=5.0, samplerate=44100, full_scale_volts=2):
    frames = int(duration * samplerate)
    rec = sd.rec(frames, samplerate=samplerate, channels=1, dtype="float32")
    sd.wait()
    x = rec[:, channel].astype(np.float64)
    rms = float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0
    v_rms = rms * float(full_scale_volts)
    p = 20e-6 * (10.0 ** (cal_spl_db / 20.0))
    s = v_rms / p if p > 0 else 0.0
    return 20.0 * np.log10(s), x, samplerate