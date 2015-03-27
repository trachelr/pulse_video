import sys
sys.path.insert(0,'/home/trachel/Projects/pulse')
#sys.path.insert(0,'/Users/henri/GitHub/pulse_video')
import cv2, time, scipy, sklearn
from scipy import interpolate, signal, fftpack
from sklearn.decomposition import PCA
import lk_tracker

import numpy as np
import pylab as plt
plt.interactive(True)


pulse = lk_tracker.PulseTracker(-1)

# 10 seconds tracking
pulse.track_len = 32*10
pulse.run()

# compute length of the tracks
ltracks = np.array([len(x) for x in pulse.tracks])
# length to keep
lkeep   = 50
# number of track keeped
nkeep   = sum(ltracks > lkeep)
# select track of length >= 101
sel_tracks = np.zeros([nkeep, lkeep])
for i in(ltracks > lkeep).nonzero()[0]:
    track = np.array(pulse.tracks[i])
    sel_tracks[i] = track[-lkeep:, 1]

t = np.linspace(0, lkeep/pulse.fps, lkeep)

# frequency sampling
fs = 250.
fs = 32.
# filter order
order = 5
# freq start 
fstart = 0.75 # ie 45 bpm
fstop  = 5 # >> 120 bpm
(b, a) = signal.butter(order, np.array([fstart, fstop])*2*np.pi/fs, 
                             btype='bandpass')

# interpolate at fs and filter
tnew = np.linspace(0, lkeep/pulse.fps, int(fs*lkeep/pulse.fps))
tracks = np.zeros([nkeep, int(fs*lkeep/pulse.fps)])
for i in range(nkeep):
    y = sel_tracks[i]
    #ynew  = interpolate.spline(t, y, tnew)
    yfilt = signal.filtfilt(b, a, y)
    tracks[i] = yfilt

# computing PCA decomposition
pca = PCA(n_components=10)
pca_tracks = pca.fit_transform(tracks.T)

# computing fourier transform of the components
freqs = fftpack.fftfreq(pca_tracks.shape[0], 1/fs)
fft_tracks = np.zeros([pca.n_components, sum(freqs > 0)])
for i in range(pca.n_components):
    fft = fftpack.fft(pca_tracks[:,i])
    fft_tracks[i] = np.abs(fft[freqs > 0])

freqs = freqs[freqs > 0]
plt.plot(freqs[freqs < 20], fft_tracks[:, freqs < 20].T)
plt.legend(['pca%i' %(i+1) for i in range(pca.n_components)])





