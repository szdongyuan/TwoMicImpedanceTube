import numpy as np
import matplotlib.pyplot as plt
from mic_cal import calibrate_mic


def main():
    sens_db, y, sr = calibrate_mic(cal_spl_db=94, channel=0, duration=5.0, samplerate=44100, full_scale_volts=2)
    print(sens_db)
    t = np.linspace(0, len(y) / sr, len(y), endpoint=False)
    Y = np.fft.rfft(y)
    f = np.fft.rfftfreq(len(y), 1 / sr)
    plt.figure(figsize=(8, 5))
    plt.subplot(2, 1, 1)
    plt.plot(t, y)
    plt.subplot(2, 1, 2)
    plt.semilogx(f, np.abs(Y))
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    main()


