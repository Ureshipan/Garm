import pyaudio
import struct
import math
import IPython.display
from ipywidgets import interact, interactive, fixed
import librosa
import librosa.display

# Packages we're using
import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.io import wavfile
from scipy.signal import butter, lfilter
import scipy.ndimage
from scipy.fft import *
import librosa.display
from scipy.fftpack import fft

INITIAL_TAP_THRESHOLD = 0.001
FORMAT = pyaudio.paInt32
SHORT_NORMALIZE = (1.0 / 32768.0)
CHANNELS = 1
RATE = 44100
INPUT_BLOCK_TIME = 0.25
INPUT_FRAMES_PER_BLOCK = int(RATE * INPUT_BLOCK_TIME)
# if we get this many noisy blocks in a row, increase the threshold
OVERSENSITIVE = 10.0 / INPUT_BLOCK_TIME
# if we get this many quiet blocks in a row, decrease the threshold
UNDERSENSITIVE = 100.0 / INPUT_BLOCK_TIME
# if the noise was longer than this many blocks, it's not a 'tap'
MAX_TAP_BLOCKS = 0.15 / INPUT_BLOCK_TIME
### Parameters ###
fft_size = 2048  # window size for the FFT
step_size = fft_size // 32  # distance to slide along the window (in time)
spec_thresh = 3  # threshold for spectrograms (lower filters out more noise)
lowcut = 200  # Hz # Low cut for our butter bandpass filter
highcut = 12000  # Hz # High cut for our butter bandpass filter
# For mels
n_mel_freq_components = 32  # number of mel frequency channels
shorten_factor = 10  # how much should we compress the x-axis (time)
start_freq = 40  # Hz # What frequency to start sampling our melS from
end_freq = 12000  # Hz # What frequency to stop sampling our melS from

win_s = 4096
hop_s = 512


def freq(data):
    # Open the file and convert to mono
    # Fourier Transform
    sample_rate = RATE
    length = len(data)
    counter = 0
    for i in range(length - 1):
        if data[i] < 0 and data[i + 1] > 0:
            counter += 1
    return counter / length


def spectral_properties(y: np.ndarray, fs: int) -> dict:
    spec = np.abs(np.fft.rfft(y))
    freq = np.fft.rfftfreq(len(y), d=1 / fs)
    spec = np.abs(spec)
    amp = spec / spec.sum()
    mean = (freq * amp).sum()
    sd = np.sqrt(np.sum(amp * ((freq - mean) ** 2)))
    amp_cumsum = np.cumsum(amp)
    median = freq[len(amp_cumsum[amp_cumsum <= 0.5]) + 1]
    mode = freq[amp.argmax()]
    Q25 = freq[len(amp_cumsum[amp_cumsum <= 0.25]) + 1]
    Q75 = freq[len(amp_cumsum[amp_cumsum <= 0.75]) + 1]
    IQR = Q75 - Q25
    z = amp - amp.mean()
    w = amp.std()
    skew = ((z ** 3).sum() / (len(spec) - 1)) / w ** 3
    kurt = ((z ** 4).sum() / (len(spec) - 1)) / w ** 4

    result_d = {
        'mean': mean,
        'sd': sd,
        'median': median,
        'mode': mode,
        'Q25': Q25,
        'Q75': Q75,
        'IQR': IQR,
        'skew': skew,
        'kurt': kurt
    }

    return result_d


def get_rms(block):
    # RMS amplitude is defined as the square root of the
    # mean over time of the square of the amplitude.
    # so we need to convert this string of bytes into
    # a string of 16-bit samples...

    # we will get one short out for each
    # two chars in the string.
    count = len(block) / 2
    format = "%dh" % (count)
    shorts = struct.unpack(format, block)

    # iterate over the block.
    sum_squares = 0.0
    for sample in shorts:
        # sample is a signed short in +/- 32768.
        # normalize it to 1.0
        n = sample * SHORT_NORMALIZE
        sum_squares += n * n

    return math.sqrt(sum_squares / count)


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def overlap(X, window_size, window_step):
    """
    Create an overlapped version of X
    Parameters
    ----------
    X : ndarray, shape=(n_samples,)
        Input signal to window and overlap
    window_size : int
        Size of windows to take
    window_step : int
        Step size between windows
    Returns
    -------
    X_strided : shape=(n_windows, window_size)
        2D array of overlapped X
    """
    if window_size % 2 != 0:
        raise ValueError("Window size must be even!")
    # Make sure there are an even number of windows before stridetricks
    append = np.zeros((window_size - len(X) % window_size))
    X = np.hstack((X, append))

    ws = window_size
    ss = window_step
    a = X

    valid = len(a) - ws
    nw = (valid) // ss
    out = np.ndarray((nw, ws), dtype=a.dtype)

    for i in np.arange(nw):
        # "slide" the window along the samples
        start = i * ss
        stop = start + ws
        out[i] = a[start:stop]

    return out


def stft(
        X, fftsize=128, step=65, mean_normalize=True, real=False, compute_onesided=True
):
    """
    Compute STFT for 1D real valued input X
    """
    x = X
    if real:
        local_fft = np.fft.rfft
        cut = -1
    else:
        local_fft = np.fft.fft
        cut = None
    if compute_onesided:
        cut = fftsize // 2
    if mean_normalize:
        print(X.mean())
        y = x - X.mean()

    y = overlap(x, fftsize, step)

    size = fftsize
    win = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(size) / (size - 1))
    y = y * win[None]
    y = local_fft(y)[:, :150]
    return y


def pretty_spectrogram(d, log=True, thresh=5, fft_size=512, step_size=64):
    """
    creates a spectrogram
    log: take the log of the spectrgram
    thresh: threshold minimum power for log spectrogram
    """
    specgram = np.abs(
        stft(d, fftsize=fft_size, step=step_size, real=False, compute_onesided=True)
    )

    if log == True:
        specgram /= specgram.max()  # volume normalize to max 1
        specgram = np.log10(specgram)  # take log
        specgram[
            specgram < -thresh
            ] = -thresh  # set anything less than the threshold as the threshold
    else:
        specgram[
            specgram < thresh
            ] = thresh  # set anything less than the threshold as the threshold

    return specgram


def av(ind, spis):
    chisl = znam = 0
    for i in ind:
        if int(i) < 10975:
            if spis[int(i)] > 100000:
                chisl += i * spis[int(i)]
                znam += spis[int(i)]
                #print(i, spis[int(i)])
    if znam != 0:
        return chisl / znam
    else:
        return 0


def pick_peaks(arr):
    posPeaks = {
        "pos": [],
        "peaks": [],
    }
    startFound = False
    n = 0
    while startFound == False:
        if arr[n] == arr[n + 1]:
            n += 1
        else:
            startFound = True

    endFound = False
    m = len(arr) - 1
    while endFound == False:
        if arr[m] == arr[m - 1]:
            m -= 1
        else:
            endFound = True

    for i in range(n + 1, m):
        if arr[i] == arr[i - 1]:
            None
        elif arr[i] >= arr[i - 1] and arr[i] >= arr[i + 1]:
            posPeaks["pos"].append(i)
            posPeaks["peaks"].append(arr[i])

    return posPeaks


class TapTester(object):
    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.stream = self.open_mic_stream()
        self.tap_threshold = INITIAL_TAP_THRESHOLD
        self.noisycount = MAX_TAP_BLOCKS + 1
        self.quietcount = 0
        self.errorcount = 0
        for i in range(self.pa.get_device_count()):
            devinfo = self.pa.get_device_info_by_index(i)
            print("Device %d: %s" % (i, devinfo["name"]))

    def stop(self):
        self.stream.close()

    def find_input_device(self):
        device_index = None
        for i in range(self.pa.get_device_count()):
            devinfo = self.pa.get_device_info_by_index(i)
            print("Device %d: %s" % (i, devinfo["name"]))

            for keyword in ["mic", "input"]:
                if keyword in devinfo["name"].lower():
                    print("Found an input: device %d - %s" % (i, devinfo["name"]))
                    device_index = i
                    return device_index

        if device_index == None:
            print("No preferred input found; using default input device.")

        return device_index

    def open_mic_stream(self):
        device_index = 1  # self.find_input_device()

        stream = self.pa.open(format=FORMAT,
                              channels=CHANNELS,
                              rate=RATE,
                              input=True,
                              input_device_index=device_index,
                              frames_per_buffer=INPUT_FRAMES_PER_BLOCK)

        return stream

    def tapDetected(self):
        print("Tap!")

    def listen(self):
        try:
            block = self.stream.read(INPUT_FRAMES_PER_BLOCK)
        except IOError as e:
            # dammit.
            self.errorcount += 1
            print("(%d) Error recording: %s" % (self.errorcount, e))
            self.noisycount = 1
            return

        amplitude = get_rms(block)
        if amplitude > self.tap_threshold:
            # noisy block

            data = np.frombuffer(self.stream.read(INPUT_FRAMES_PER_BLOCK),
                                 dtype=np.int16)  # butter_bandpass_filter(block, lowcut, highcut, RATE, order=1)
            # print(data.decode())
            data = butter_bandpass_filter(data, lowcut, highcut, RATE, order=1)[100:]

            # b = [(ele / 2 ** 32.) * 2 - 1 for ele in data]  # this is 8-bit track, b is now normalized on [-1,1)
            # c = fft(b)  # calculate fourier transform (complex numbers list)
            # d = int(len(c) / 2) # you only need half of the fft list (real signal symmetry)
            # plt.plot(fft(data), 'r')
            # #plt.plot(data)
            # plt.xlim([0, 1000])
            # plt.show()

            fs_rate, signal = RATE, data
            # print("Frequency sampling", fs_rate)
            l_audio = len(signal.shape)
            # print("Channels", l_audio)
            if l_audio == 2:
                signal = signal.sum(axis=1) / 2
            N = signal.shape[0]
            # print("Complete Samplings N", N)
            secs = N / float(fs_rate)
            # print("secs", secs)
            Ts = 1.0 / fs_rate  # sampling interval in time
            # print("Timestep between samples Ts", Ts)
            t = np.arange(0, secs, Ts)  # time vector as scipy arange field / numpy.ndarray
            FFT = abs(fft(signal))
            FFT_side = FFT[range(N // 2)]  # one side FFT range
            freqs = scipy.fftpack.fftfreq(signal.size, t[1] - t[0])
            fft_freqs = np.array(freqs)
            freqs_side = freqs[range(N // 2)]  # one side frequency range
            fft_freqs_side = np.array(freqs_side)
            plt.subplot(311)
            p1 = plt.plot(t, signal, "g")  # plotting the signal
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.subplot(312)
           # print(fft_freqs[:10])
            #print(FFT_side[:10])
            fftsi = FFT_side#np.hstack([FFT_side, FFT_side[::]])
            #print(len(fftsi), len(fft_freqs))
            p3 = plt.plot(abs(fft_freqs[:len(fft_freqs) // 2] - 10974), abs(fftsi), "b")  # plotting the positive fft spectrum
            plt.xlabel('Frequency (Hz)')

            plt.ylabel('Count single-sided')
            #print(abs(fft_freqs[:len(fft_freqs) // 2] - 10974))
            print(av(abs(fft_freqs[:len(fft_freqs) // 2] - 10974), abs(fftsi)))
            # plt.xlim([50, 1000])
            plt.show()
            wait = input()

            # fig, ax = plt.subplots()
            # img = librosa.display.specshow(librosa.amplitude_to_db(data**2, ref=np.max),
            #                               y_axis='log', x_axis='time', ax=ax)
            # ax.set_title('Power spectrogram')
            # fig.colorbar(img, ax=ax, format="%+2.0f dB")

            # wav_spectrogram = pretty_spectrogram(
            #     data,#.astype("float64"),
            #     fft_size=fft_size,
            #     step_size=step_size,
            #     log=True,
            #     thresh=spec_thresh,
            # )
            #
            # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(1, 10))#(round(np.shape(data)[0] / float(RATE), 0), 1))
            # ax = ax.matshow(
            #     np.transpose(wav_spectrogram),
            #     interpolation="nearest",
            #     aspect="auto",
            #     cmap=plt.cm.gray,
            #     origin="lower",
            # )
            # print(fig)
            # fig.patch.set_visible(False)
            # plt.axis('off')
            # ax.spines['top'].set_visible(False)
            # ax.spines['right'].set_visible(False)
            # ax.spines['bottom'].set_visible(False)
            # ax.spines['left'].set_visible(False)
            # plt.ylim([40, 120])
            # plt.show()
            # plt.show(block=False)

            # print('saving')
            # plt.savefig('temp.png')
            # plt.clf()
            # plt.close(fig)

            self.quietcount = 0
            self.noisycount += 1
            if self.noisycount > OVERSENSITIVE:
                # turn down the sensitivity
                self.tap_threshold *= 1.1
        else:
            # quiet block.

            if 1 <= self.noisycount <= MAX_TAP_BLOCKS:
                self.tapDetected()
            self.noisycount = 0
            self.quietcount += 1
            if self.quietcount > UNDERSENSITIVE:
                # turn up the sensitivity
                self.tap_threshold *= 0.9


if __name__ == "__main__":
    tt = TapTester()

    for i in range(1000):
        tt.listen()
