import sys
sys.path.insert(0,'/home/trachel/Projects/pulse_video')
import cv2, time, scipy
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

video_name = video_list[1]

pulse = lk_tracker.PulseTracker(video_path + video_name)

# 10 seconds tracking
pulse.track_len = 32*10
pulse.run()

# compute length of the tracks
ltracks = np.array([len(x) for x in pulse.tracks])

# get first tracked point (the longest)
y = np.array(pulse.tracks[0])
nspl = y.shape[0]
t = np.linspace(0, nspl/32., nspl)

fs = 250.
# interpolate at fs
tnew = np.linspace(0, nspl/32., int(nspl*fs))
ynew = scipy.interpolate.spline(t ,y[:,1], tnew)

# filter order
order = 5
# freq start 
fstart = 0.75 # ie 45 bpm
fstop  = 5 # >> 120 bpm

(b, a) = scipy.signal.butter(order, np.array([fstart, fstop])*2*np.pi/fs, 
                             btype='bandpass')

yfilt = scipy.signal.filtfilt(b,a,ynew)

