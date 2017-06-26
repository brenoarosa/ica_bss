# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from glob import glob
from sklearn.decomposition import FastICA
from mutual_info import mutual_information_2d
from plot_confusion import plot_confusion_matrix
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


def mutual_info_pairwise(matrix, normalize=True):
    """
    Calculates mutual information between matrix columns.
    Normalization is required to valid results.

    If normalize parameters is True, mormalize matrix to 16-bits integer.
    Scaling it to be 1/2 of it's dynamic range.
    """
    size = matrix.shape[1]

    if normalize:
        factor = 2.**14/np.max(np.abs(matrix), axis=0) * np.eye(size)
        matrix = np.dot(matrix, factor).astype('int16')

    mi_matrix = np.zeros(shape=(size, size), dtype="float64")
    for i in range(size):
        for j in range(i, size):
            mi = mutual_information_2d(matrix[:, i], matrix[:, j])
            mi = 1 - np.exp(-2 * mi)
            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi

    return mi_matrix


SOURCE_GLOB = "mono/*.wav"
SOURCES_COUNT = len(glob(SOURCE_GLOB))
MICROPHONE_COUNT = SOURCES_COUNT
MICROPHONE_COUNT = 5
audio_len = int(5e5) # .5M samples

if not os.path.exists('output'):
    os.makedirs('output')

assert(MICROPHONE_COUNT >= SOURCES_COUNT)


# Source matrix
S = np.zeros(shape=(audio_len, SOURCES_COUNT), dtype='int16')
for i, filename in enumerate(glob(SOURCE_GLOB)):
    sr, s = wavfile.read(filename)
    S[:, i] = s[0:audio_len]

# Mixing matrix
A = np.random.rand(MICROPHONE_COUNT, SOURCES_COUNT)
A = (A * .8) + .1 # limits from .1 to .9
print('Mixing matrix: \n{}'.format(A))

X = np.dot(S, A.T)  # Generate observations

# Compute ICA
ica = FastICA(n_components=SOURCES_COUNT)
S_ = ica.fit_transform(X)  # Reconstruct signals
A_ = ica.mixing_  # Get estimated mixing matrix


# Saving mixed and recovered sources to disk
for i in range(X.shape[1]):
    save_audio(X[:, i], sr, 'output/mixed_{}.wav'.format(i))

for i in range(S_.shape[1]):
    save_audio(S_[:, i], sr, 'output/{}.wav'.format(i))


# MI matrixs
mi_S = mutual_info_pairwise(S, False)
print("Sources Mutual Info:")
print(mi_S)

mi_X = mutual_info_pairwise(X, False)
print("Microphones Mutual Info:")
print(mi_X)

mi_S_ = mutual_info_pairwise(S_, False)
print("Reconstructed Sources Mutual Info:")
print(mi_S_)


plot_confusion_matrix(mi_S, 'mi_S.png', title='Informacao Mutua das Fontes')
plot_confusion_matrix(mi_X, 'mi_X.png', title='Informacao Mutua das Misturas')
plot_confusion_matrix(mi_S_, 'mi_S_.png', title='Informacao Mutua das Reconstituicoes')
