import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import wave as we

sample_rate, data = wavfile.read('test_noise.wav')
t = int(len(data)/sample_rate)
Fs = t*sample_rate
print('sample rate=', sample_rate)
print('sample=', len(data))
print('time=', len(data)/sample_rate)

n = np.linspace(0, t, Fs)
plt.plot(n, data)
plt.xlabel('Time(s)')
plt.ylabel('Amplitude')
plt.title('Noisy Signal Waveforms')
plt.show()


def moving_avg(audio_data, avg_length):
    new_data = []
    for i in range(data.shape[0]-avg_length+1):
        new_data.append(np.average(data[i:i+avg_length]))
    new_data = np.array(new_data).reshape((-1, 1))
    new_data = np.int32(new_data)
    return new_data


length = data.shape[0]/16000
for i in range(80,100,10):
    new_data = moving_avg(data, i)
    time = np.linspace(0., length, new_data.shape[0])
    plt.plot(time, new_data)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Moving Avg Filter  "+str(i))
    plt.show()

    save_wav = new_data.real.reshape((len(new_data), 1)).T.astype(np.short)
    f = we.open("1)moving_avg_fliter.wav", "wb")
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(16000)
    f.writeframes(save_wav.tostring())
    f.close()
