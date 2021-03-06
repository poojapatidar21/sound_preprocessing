#new code
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import wave as we

sample_rate, data = wavfile.read('test_noise.wav')
t = int(len(data)/sample_rate)
Fs = t*sample_rate
n = np.linspace(0, t, Fs)


def fft_wav(waveData):
    f_array = np.fft.fft(waveData)
    f_abs = f_array
    axis_f = np.linspace(0, 100, np.int(len(f_array)))
    plt.plot(axis_f, np.abs(f_abs[0:len(axis_f)]))
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude spectrum")
    plt.title("Tile map")
    plt.show()
    return f_abs


wave_fft = fft_wav(data)
step_hz = 100/(len(data)/2)
tab_hz = 68
new_wav = wave_fft.copy()

for i in range(int(tab_hz/step_hz), (len(wave_fft) - int(tab_hz/step_hz))):
    new_wav[i] = 0

axis_f = np.linspace(0, 100, np.int(len(wave_fft)))
plt.plot(axis_f, np.abs(new_wav[0:len(axis_f)]))
plt.xlabel("Frequency")
plt.ylabel("Amplitude spectrum")
plt.title("Tile map after wave filtering")
plt.show()

filtered_wave = np.fft.ifft(new_wav)
plt.plot(n, filtered_wave.real)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Single channel wavedata after wave filtering")
plt.show()

save_wav = filtered_wave.real.reshape((len(filtered_wave), 1)).T.astype(np.short)
f = we.open("2)freq_flitering.wav", "wb")
f.setnchannels(1)
f.setsampwidth(2)
f.setframerate(16000)
f.writeframes(save_wav.tostring())
f.close()
