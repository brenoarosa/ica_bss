import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from glob import glob
from sklearn.decomposition import FastICA
import os


def save_audio(x, sample_rate=44100, filepath='audio.wav'):
    """
    Save audio to disk.

    Normalize audio to 16-bits integer.
    Scale it to be 1/2 of it's dynamic range.
    """
    x_o = x # copy to preserve original signal
    factor = 2.0**14/np.max(np.abs(x))
    x_o = np.asarray(x_o*factor, dtype="int16")
    wavfile.write(filepath, sr, x_o)


SOURCE_GLOB = "mono/*.wav"
SOURCES_COUNT = len(glob(SOURCE_GLOB))
MICROPHONE_COUNT = SOURCES_COUNT
audio_len = int(6e6) # 6M samples

if not os.path.exists('output'):
    os.makedirs('output')

assert(MICROPHONE_COUNT >= SOURCES_COUNT)

# Source matrix
S = np.zeros((audio_len, SOURCES_COUNT))
for i, filename in enumerate(glob(SOURCE_GLOB)):
    sr, s = wavfile.read(filename)
    S[:, i] = s[0:audio_len]


# Mixing matrix
A = np.random.rand(MICROPHONE_COUNT, SOURCES_COUNT)
print('Mixing matrix: \n{}'.format(A))

X = np.dot(S, A.T)  # Generate observations

for i in range(X.shape[1]):
    save_audio(X[:, i], sr, 'output/mixed_{}.wav'.format(i))

# Compute ICA
ica = FastICA(n_components=SOURCES_COUNT)
S_ = ica.fit_transform(X)  # Reconstruct signals
A_ = ica.mixing_  # Get estimated mixing matrix

for i in range(S_.shape[1]):
    save_audio(S_[:, i], sr, 'output/{}.wav'.format(i))


"""
# Plot results

plt.figure()

models = [X, S, S_]
names = ['Observations (mixed signal)',
         'True Sources',
         'ICA recovered signals']
colors = ['red', 'steelblue', 'orange']

for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(4, 1, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)

plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
plt.show()
"""
