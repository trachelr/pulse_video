'''
Lucas-Kanade tracker
====================
Lucas-Kanade sparse optical flow application for detecting pulse from head motion. 
Uses goodFeaturesToTrack for track initialization and back-tracking for match verification
between frames.

Inspired from opencv_source_code/samples/python2/lk_track.py

-----
Author : Romain Trachel <romain.trachel@ens.fr>
Date   : 02/19/2015

'''

import numpy as np
import cv2
from time import clock
from scipy import interpolate, signal, fftpack
from sklearn.decomposition import PCA

import numpy as np
import pylab as plt
#plt.interactive(True)

# hard coded parameters (beark... some of them need to be passed into init)
# parameters of the Lucas-Kanade optical flow algorithm
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# parameters of the feature tracking algorithm
feature_params = dict(maxCorners = 500,
                      qualityLevel = 0.05,  # decrease sensitivity
                      minDistance = 7,
                      blockSize = 7 )

# parameters of the face tracking algorithm
face_params = dict(scaleFactor=1.1, 
                   minNeighbors=5,
                   minSize=(30, 30),
                   flags=cv2.cv.CV_HAAR_SCALE_IMAGE)

cv_path = '/Users/trachel.r/anaconda/pkgs/opencv-2.4.8-np17py27_2/share/OpenCV/haarcascades/'
# cv_path = np.loadtxt('cv_path.conf')
face_cascade = cv2.CascadeClassifier(cv_path + 'haarcascades/haarcascade_frontalface_default.xml')

class PulseTracker:
    
    '''
        
        Main class of the pulse detection project
        
        Parameters    
        ----------
            video_src : string, int
                source of the video, if already recorded then provide the
                full path of the video, else provide an int to use a webcam.
                (see Capture Video from Camera in the OpenCV-Python tutorial)
                
            fps : int (default, 32)
                frame per seconds
            
            crop_h : float (default, 0.45)
                percentage of face cropping on the forehead
                45% takes the nose and the mouth
        
        Attributes
        ----------
            tracks : list
                coordinates of the tracked points for each frames
            track_len : int
                number of time samples (frames) to keep in tracks
            capture : VideoCapture
                an instance of openCV providing video from the camera
            frame_idx : int
                index of the current frame
            face : (int, int, int, int)
                face rectangle (x, y, height, width)
    '''
    def __init__(self, video_src, fps=32., crop_h=.45):
        self.track_len = 10
        self.detect_interval = 5
        self.beat_interval = 20
        self.tracks  = []
        self.capture = cv2.VideoCapture(video_src)
        self.frame_idx  = 0
        self.face = ()
        # try to guess fps from the camera first
        if (video_src == -1) and (cap.get(cv2.cv.CV_CAP_PROP_FPS) != 0):
            self.fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        else:
            self.fps = fps
        
        self.crop_h = crop_h
        # ------
        # signal processing attributes :
        # ------
        # frequency sampling
        self.pulse_fs = 250.
        # filter order
        self.ford = 5
        # freq start 
        self.fstart = 0.75 # ie 45 bpm
        self.fstop  = 5 # >> 120 bpm
        # normalized frequency
        wn = np.array([self.fstart, self.fstop])*2*np.pi/self.pulse_fs
        # define filter coefs
        (b, a) = signal.butter(self.ford, wn, btype='bandpass')
        # and make it attributes
        self._filter_b, self._filter_a = b, a
        # create PCA
        self.pulse_pca = PCA(n_components=10)
    
    def crop_face(self, face):
        fh = int(face[3] * self.crop_h)
        # remove the forehead and crop the face tracking rectangle
        return face[0]+5, face[1]+fh, face[2]-10, face[3]-fh
    
    def run(self):
        while self.capture.isOpened():
            # getting a frame
            ret, frame = self.capture.read()
            # convert into gray scale
            f0_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # make a copy for visualization
            vis = frame.copy()
            # apply face detection
            faces = face_cascade.detectMultiScale(f0_gray, **face_params)
            
            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, f0_gray
                # get tracking points
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                # apply Lucas-Kanade tracker with optical flow
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1
                new_tracks = []
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks.append(tr)
                    cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
                
                self.tracks = new_tracks
                cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
                print 'track count: %d' % len(self.tracks)
            
            
            if self.frame_idx % self.detect_interval == 0:
                # find the face
                faces = face_cascade.detectMultiScale(f0_gray, **face_params)
                if len(faces) > 0:
                    # get face coordinates
                    (x, y, w, h) = self.crop_face(faces[0])
                    cv2.rectangle(vis, (x,y), (x+w,y+h),(0,255,0),2)
                    
                    mask = np.zeros_like(f0_gray)
                    mask[x:x+h, y:y+w] = 255
                    for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                        cv2.circle(mask, (x, y), 5, 0, -1)
                    # Get the good features to track in the face
                    p = cv2.goodFeaturesToTrack(f0_gray, mask = mask, **feature_params)
                    if p is not None:
                        for x, y in np.float32(p).reshape(-1, 2):
                            self.tracks.append([(x, y)])
            
            # compute length of the tracks
            ltracks = np.array([len(x) for x in self.tracks])
            # length to keep
            lkeep   = self.track_len
            # number of track keeped
            nkeep   = sum(ltracks > lkeep)
            # compute pulse rate
            if self.frame_idx % self.beat_interval == 0 and nkeep > 0:
                # select track of length >= 101
                tracks = np.zeros([nkeep, lkeep])
                for i in(ltracks == lkeep).nonzero()[0]:
                    track = np.array(self.tracks[i])
                    # keep only y coordinates !!!
                    tracks[i] = track[:, 1]

                pca_tracks = self.pulse_pca.fit_transform(tracks.T)

                # computing fourier transform of the components
                freqs = fftpack.fftfreq(tnew.shape[0], 1/fs)
                fft_tracks = np.zeros([pca.n_components, sum(freqs > 0)])
                for i in range(pca.n_components):
                    fft = fftpack.fft(pca_tracks[:,i])
                    fft_tracks[i] = np.abs(fft[freqs > 0])

                freqs = freqs[freqs > 0]
                plt.plot(freqs[freqs < 20], fft_tracks[:, freqs < 20].T)
                plt.legend(['pca%i' %(i+1) for i in range(pca.n_components)])
                
            
            self.frame_idx += 1
            self.prev_gray = f0_gray
            cv2.imshow('lk_track', vis)

            ch = 0xFF & cv2.waitKey()
            if ch == 27:
                break
    
    def close(self):
        self.capture.release()

def main():
    
    try:
        # start tracking with the webcam
        pulse = PulseTracker(-1)
        pulse.run()
    except KeyboardInterrupt:
        pulse.close()
        cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
