import librosa
import matplotlib.pyplot as plt
import numpy as np

audio_file = 'theave.mp3'

y, sr = librosa.load(audio_file)
mfccs = librosa.feature.mfcc(y=y, sr=sr)
delta_mfcc = librosa.feature.delta(mfccs)
delta2_mfcc = librosa.feature.delta(mfccs, order=2)

mfcc_vector = mfccs[:14]
delta_vector = delta_mfcc[:14]
delta2_vector = delta2_mfcc[:14]

features = np.concatenate([mfcc_vector, delta_vector, delta2_vector])
