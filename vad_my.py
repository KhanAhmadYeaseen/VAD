import numpy as np
import matplotlib.pyplot as plt
import librosa

plt.style.use('ggplot')

file_path = 'data/c5_Suzuki-chest.wav'
output_file_path = 'data/z.txt'


def print_label(bin, window_len):
    file_out = open(output_file_path, "w")

    in_speech_flag = 0
    for i in range(0, len(bin)):
        if bin[i] == 1 and in_speech_flag == 0:
            file_out.write(f'{(i * window_len):.6f}')
            file_out.write('\t')
            in_speech_flag = 1
        if bin[i] == 0 and in_speech_flag == 1:
            file_out.write(f'{(i * window_len):.6f}')
            file_out.write('\t')
            in_speech_flag = 0
            file_out.write("speech\n")

    if in_speech_flag == 1:
        file_out.write(f'{((len(bin) - 1) * window_len):.6f}')
        file_out.write('\t')
        file_out.write("speech\n")


def fft_window(audio_data, sampling_r, high_cut, low_cut, window_l, window_num):
    time_s = 1.0 / sampling_r

    time_start = 0.000 + (window_num - 1) * window_l
    time_end = 0.000 + window_num * window_l

    time_start_in = int(time_start * sampling_r)
    time_end_in = int(time_end * sampling_r)

    y = audio_data[time_start_in:time_end_in]
    han = np.hanning(len(y))
    y = y * han
    len_y = len(y)

    am = 2 / len_y * np.fft.fft(y)
    am = np.abs(am)

    fr = np.fft.fftfreq(len(am), time_s)

    fr_xr = fr[0:int(len(fr) / 2)]
    am_yr = am[0:int(len(fr) / 2)]

    hi_in = len(fr_xr) - 1
    hi_fr = fr_xr[hi_in]

    hi_cut_in = int(high_cut * (hi_in / hi_fr))
    lo_cut_in = int(low_cut * (hi_in / hi_fr))

    fr_xr = fr_xr[lo_cut_in:hi_cut_in]
    am_yr = am_yr[lo_cut_in:hi_cut_in]

    return fr_xr, am_yr


y, sampling_rate = librosa.load(file_path, sr=None, mono=True, offset=0.0)
time_step = 1.0 / sampling_rate
hi_cut = 3000
lo_cut = 250
window_len = 0.03
window_number = 1
amplitude_threshold_param = 0.05
total_time = len(y) / sampling_rate
max_window_num = total_time / window_len

print("total time : " + str(total_time))
print("window length : " + str(window_len))
print("max window num : " + str(max_window_num))

amplitude_final = []
for i in range(1, int(max_window_num) + 1):
    fr_x, am_y = fft_window(y, sampling_rate, hi_cut, lo_cut, window_len, i)
    amplitude_final.append(np.sum(am_y))

plt.ylim(0, 1.5)
plt.plot(amplitude_final)
plt.show()

bin_final = []
for i in amplitude_final:
    if i > amplitude_threshold_param:
        bin_final.append(1)
    else:
        bin_final.append(0)

print(amplitude_final)
plt.ylim(0, 1.5)
plt.plot(bin_final)
plt.show()

print_label(bin_final, window_len)

print("maximum amp : " + str(max(amplitude_final)))
