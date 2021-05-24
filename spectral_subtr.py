import scipy.io.wavfile as wav
import numpy as np
import matplotlib.pyplot as plt

file = 'C:\\Users\\Pooja Patidar\\Desktop\\Sem 4\\Data Handling\\assignment 2\\test_noise.wav'
sr, data = wav.read(file)
fl = 400  # frame_length
frames = []
for i in range(0, int(len(data)/(int(fl/2))-1)):
    arr = data[int(i*int(fl/2)):int(i*int(fl/2)+fl)]
    frames.append(arr)

frames = np.array(frames)
ham_window = np.hamming(fl)
windowed_frames = frames*ham_window

dft = []
for i in windowed_frames:
    dft.append(np.fft.fft(i))

dft = np.array(dft)
dft_mag_spec = np.abs(dft)
dft_phase_spec = np.angle(dft)
noise_estimate = np.mean(dft_mag_spec, axis=0)
noise_estimate_mag = np.abs(noise_estimate)
estimate_mag = (dft_mag_spec-2*noise_estimate_mag)
estimate_mag[estimate_mag < 0] = 0
estimate = estimate_mag*np.exp(1j*dft_phase_spec)

ift = []
for i in estimate:
    ift.append(np.fft.ifft(i))

clean_data = []
clean_data.extend(ift[0][:int(fl/2)])
for i in range(len(ift)-1):
    clean_data.extend(ift[i][int(fl/2):]+ift[i+1][:int(fl/2)])

clean_data.extend(ift[-1][int(fl/2):])
clean_data = np.array(clean_data)

plt.plot(np.linspace(0, 64000, 64000), data)
plt.plot(np.linspace(0, 64000, 64000), clean_data)

cleaned_file = "3)Spectral_sub.wav"
wav.write(cleaned_file, rate=sr, data=clean_data.astype(np.int16))
