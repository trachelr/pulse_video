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
import time
from scipy import interpolate, signal, fftpack, optimize
from sklearn.decomposition import PCA
import pylab as plt
plt.interactive(True)

# hard coded parameters (beark... some of them need to be passed into init)
# parameters of the Lucas-Kanade optical flow algorithm
lk_params = dict( winSize  = (35, 35),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# parameters of the feature tracking algorithm
feature_params = dict(maxCorners = 500,
                      qualityLevel = 0.35,  # decrease sensitivity
                      minDistance = 7,
                      blockSize = 7 )

# parameters of the face tracking algorithm
face_params = dict(scaleFactor=1.1, 
                   minNeighbors=5,
                   minSize=(80, 80),
                   flags=cv2.cv.CV_HAAR_SCALE_IMAGE)

face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')

# gaussian function used to find BPM on the FFT
def gauss_func(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

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
    def __init__(self, video_src, track_len=32, fps=10., crop_height=.45):
        self.track_len = track_len
        self.detect_interval = 10
        self.beat_interval = 20
        self.tracks  = []
        self.times   = []
        self.capture = cv2.VideoCapture(video_src)
        self.frame_idx  = 0
        self.face = ()
        # try to guess fps from the camera first
        if (video_src == -1) and (self.capture.get(cv2.cv.CV_CAP_PROP_FPS) != 0):
            self.fps = self.capture.get(cv2.cv.CV_CAP_PROP_FPS)
        else:
            self.fps = fps
        
        # height (percent) cropping
        self.crop_height = crop_height
        self.crop_width = .5
        # face position
        self.face_pos = (0, 0, 0, 0)
        # ------
        # signal processing attributes :
        # ------
        # frequency sampling
        self.pulse_fs = 250.
        # filter order
        self.ford = 6
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
        
        # ------------------------
        #  plot fft + gaussian
        # ------------------------
        fig  = plt.figure()
        plt.xlabel('pulse/min')
        plt.ylabel('power spectral density')
        axes = fig.add_subplot(111)
        axes.set_autoscale_on(True)
        axes.autoscale_view(True, True, True)
        freqs = fftpack.fftfreq(self.track_len, 1/(self.fps*1.))
        freqs = freqs[freqs > 0]
        # take lower frequency and convert in BPM
        freqs = freqs[freqs < 5]*60 
        self.pca_lines = [axes.plot(freqs, np.zeros_like(freqs))[0] for a in range(5)]
        #self.gau_line, = axes.plot(freqs, np.zeros_like(freqs), 'b--')
        
    
    def crop_top_face(self, face):
        fh = int(face[3] * self.crop_height)
        fw = int(face[2] * self.crop_width)
        # remove the forehead and crop the face tracking rectangle
        return face[0]+fw/2, face[1]+fh, face[2]-fw, face[3]-fh
    
    def crop_bottom_face(self, face):
        fh = int(face[3] * (self.crop_height-.25))
        fw = int(face[2] * self.crop_width)
        return face[0]+fw/2, face[1], face[2]-fw, fh
    
    def run(self):
        print 'runing calibration (waiting %.3f sec)' %(self.track_len/(1.*self.fps))
        # print times t1, t2, t3 and stop other print
        timing_debug = False
        while self.capture.isOpened():
            t0 = time.time()
            t1, t2, t3 = 0, 0, 0
            # getting a frame
            ret, frame = self.capture.read()
            # convert into gray scale
            f0_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # make a copy for visualization
            vis = frame.copy()
            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, f0_gray
                # get tracking points
                t1 = time.time() - t0
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                # apply Lucas-Kanade tracker with optical flow
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                t2 = time.time() - t0
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
                
                t3 = time.time() - t0
                self.tracks = new_tracks
                cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
            
            if self.frame_idx % self.detect_interval == 0:
                # find the face
                faces = face_cascade.detectMultiScale(f0_gray, **face_params)
                sizes = [h*w for x, y, w, h in faces]
                if len(faces) > 0:
                    sel_face = np.argmax(sizes)
                    self.face = faces[sel_face]
                    # get face coordinates
                    (x, y, w, h) = self.crop_top_face(self.face)
                    #(x, y, w, h) = self.crop_bottom_face(self.face)
                    # and save it
                    self.face_pos = (x, y, w, h)
                    # Get the good features to track in the face
                    face_gray = f0_gray[x:x+h, y:y+w]
                    p = cv2.goodFeaturesToTrack(face_gray , mask = None, **feature_params)
                    if p is not None:
                        for xx, yy in np.float32(p).reshape(-1, 2):
                            self.tracks.append([(x+xx, y+yy)])
            
            # display the face
            (x, y, w, h) = self.face_pos
            cv2.rectangle(vis, (x,y), (x+w,y+h),(0,255,0),2)
            # compute length of the tracks
            ltracks = np.array([len(x) for x in self.tracks])
            # number of track keeped
            nkeep   = sum(ltracks >= self.track_len)
            # compute pulse rate
            if self.frame_idx % self.beat_interval == 0 and nkeep > 10:
                if not timing_debug:
                    print '---  starting pulse '
                    print 'track count: %d' % len(self.tracks)
                
                # select track of length >= 101
                tracks = np.zeros([nkeep, self.track_len])
                for i in(ltracks == self.track_len).nonzero()[0]:
                    track = np.array(self.tracks[i])
                    # keep only y coordinates !!!
                    tracks[i] = signal.filtfilt(self._filter_b, self._filter_a, track[:, 1])

                pca_tracks = self.pulse_pca.fit_transform(tracks.T)

                # computing fourier transform of the components
                n_comp = self.pulse_pca.n_components
                freqs = fftpack.fftfreq(self.track_len, 1/(self.fps*1.))
                fft_tracks = np.zeros([n_comp, sum(freqs > 0)])
                for i in range(n_comp):
                    fft = fftpack.fft(pca_tracks[:,i])
                    fft_tracks[i] = np.abs(fft[freqs > 0])

                freqs = freqs[freqs > 0]
                fft_tracks = fft_tracks[:, freqs > 0]
                # fit a gaussian on the 1st component
                #try:
                #    popt, pcov = optimize.curve_fit(gauss_func, freqs, fft_tracks[0])
                #    bpm_prob = gauss_func(freqs, popt[0], popt[1], popt[2])
                #    fmax  = np.argmax(bpm_prob)
                #    self.gau_line.set_data(freqs[freqs < 20], bpm_prob[freqs < 20])
                #except RuntimeError:
                fmax  = np.argmax(fft_tracks[0])
                #self.fft_line.set_data(freqs[freqs < 5]*60, fft_tracks[0, freqs < 20].T)
                for a in range(5):
                    self.pca_lines[a].set_data(freqs[freqs < 5]*60, fft_tracks[a, freqs < 5].T)
                
                plt.legend(['pca%i' %(i+1) for i in range(n_comp)])
                plt.title('BPM %.3f' %(freqs[fmax]*60))
                plt.ylim([0, np.ceil(fft_tracks[0].max())])
            
            self.frame_idx += 1
            self.prev_gray = f0_gray
            cv2.imshow('lk_track', vis)
            
            dt = time.time() - t0
            
            islate = 1/self.fps - dt < 0
            if not islate:
                time.sleep(1/self.fps - dt)
            elif islate and not timing_debug:
                print '%.3f ms delayed' %(dt - 1/self.fps)
            elif timing_debug:
                print (len(self.tracks), t1, t2, t3)
            
            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break
    
    def close(self):
        self.capture.release()

def main():
    
    try:
        # start tracking with the webcam
        pulse = PulseTracker(-1)
        # buffer of 30sec @10fps
        pulse.track_len = 100
        # run main program
        pulse.run()
    except KeyboardInterrupt:
        pulse.close()
        cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
