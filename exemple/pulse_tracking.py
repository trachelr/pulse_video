import sys
sys.path.insert(0,'/home/trachel/Projects/pulse')
import cv2, time, scipy, sklearn
from scipy import interpolate, signal, fftpack
from sklearn.decomposition import PCA
import lk_tracker

import numpy as np
import pylab as plt
plt.interactive(True)

video_path = '/data/mindwandering/video/p01/'
video_list = ['MW_Video_Wed_Feb_18_16:11:20_2015.avi', 
              'MW_Video_Wed_Feb_18_16:19:29_2015.avi',
              'MW_Video_Wed_Feb_18_16:31:15_2015.avi', 
              'MW_Video_Wed_Feb_18_16:46:38_2015.avi',
              'MW_Video_Wed_Feb_18_16:57:10_2015.avi',
              'MW_Video_Wed_Feb_18_17:11:42_2015.avi',
              'MW_Video_Wed_Feb_18_17:27:13_2015.avi',
              'MW_Video_Wed_Feb_18_17:44:19_2015.avi']

video_name = video_list[-2]

pulse = lk_tracker.PulseTracker(video_path + video_name)

# 10 seconds tracking
pulse.track_len = 32*10
pulse.run()

# compute length of the tracks
ltracks = np.array([len(x) for x in pulse.tracks])
# length to keep
lkeep   = 319
# number of track keeped
nkeep   = sum(ltracks > lkeep)
# select track of length >= 101
sel_tracks = np.zeros([nkeep, lkeep])
for i in(ltracks > lkeep).nonzero()[0]:
    track = np.array(pulse.tracks[i])
    sel_tracks[i] = track[-lkeep:, 1]

t = np.linspace(0, lkeep/32., lkeep)

# frequency sampling
fs = 250.
# filter order
order = 5
# freq start 
fstart = 0.75 # ie 45 bpm
fstop  = 5 # >> 120 bpm
(b, a) = signal.butter(order, np.array([fstart, fstop])*2*np.pi/fs, 
                             btype='bandpass')

# interpolate at fs and filter
tnew = np.linspace(0, lkeep/32., int(fs*lkeep/32.))
tracks = np.zeros([nkeep, int(fs*lkeep/32.)])
for i in range(nkeep):
    y = sel_tracks[i]
    ynew  = interpolate.spline(t, y, tnew)
    yfilt = signal.filtfilt(b, a, ynew)
    tracks[i] = yfilt

# computing PCA decomposition
pca = PCA(n_components=10)
pca_tracks = pca.fit_transform(tracks.T)

# computing fourier transform of the components
freqs = fftpack.fftfreq(tnew.shape[0], 1/fs)
fft_tracks = np.zeros([pca.n_components, sum(freqs > 0)])
for i in range(pca.n_components):
    fft = fftpack.fft(pca_tracks[:,i])
    fft_tracks[i] = np.abs(fft[freqs > 0])

freqs = freqs[freqs > 0]
plt.plot(freqs[freqs < 20], fft_tracks[:, freqs < 20].T)
plt.legend(['pca%i' %(i+1) for i in range(pca.n_components)])





