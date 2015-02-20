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

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners = 500,
                      qualityLevel = 0.05,  # decrease sensitivity
                      minDistance = 7,
                      blockSize = 7 )

face_params = dict(scaleFactor=1.1, 
                   minNeighbors=5,
                   minSize=(30, 30),
                   flags=cv2.cv.CV_HAAR_SCALE_IMAGE)

face_cascade = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml')

class PulseTracker:
    
    def __init__(self, video_src, fps=32., crop_h=.45):
        self.track_len = 10
        self.detect_interval = 5
        self.tracks  = []
        self.capture = cv2.VideoCapture(video_src)
        self.frame_idx  = 0
        self.face = ()
        self.fps = fps
        self.crop_h = crop_h
    
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
            
            self.frame_idx += 1
            self.prev_gray = f0_gray
            cv2.imshow('lk_track', vis)

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break
    
    def close(self):
        self.capture.release()

def main():
    
    try:
        pulse = PulseTracker(-1)
        pulse.run()
    except KeyboardInterrupt:
        pulse.close()
        cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
