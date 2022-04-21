import pyaudio
import sys
from aubio import source, pitch
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

pa = pyaudio.PyAudio()

stream = pa.open(format=FORMAT,
                 channels=CHANNELS,
                 rate=RATE,
                 input=True,
                 input_device_index=1,
                 frames_per_buffer=INPUT_FRAMES_PER_BLOCK)


s = source(np.frombuffer(stream.read(INPUT_FRAMES_PER_BLOCK), dtype=np.int16), RATE, hop_s)
samplerate = s.samplerate

tolerance = 0.8

pitch_o = pitch("yin", win_s, hop_s, samplerate)
pitch_o.set_unit("midi")
pitch_o.set_tolerance(tolerance)

pitches = []
confidences = []

total_frames = 0
while True:
    samples, read = s()
    pitch = pitch_o(samples)[0]
    pitches += [pitch]
    confidence = pitch_o.get_confidence()
    confidences += [confidence]
    total_frames += read
    if read < hop_s:
        break

print("Average frequency = " + str(np.array(pitches).mean()) + " hz")