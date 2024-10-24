import cv2
import numpy as np
from matplotlib import pyplot as plt
from math import sin, cos, radians

cap = cv2.VideoCapture('output.avi')
bparams = cv2.SimpleBlobDetector_Params()
bparams.filterByArea = True
bparams.minArea = 30
bparams.maxArea = 300
bparams.filterByCircularity = True
bparams.minCircularity = 0.5
bparams.filterByConvexity = False
bparams.filterByInertia = True
bparams.minInertiaRatio = 0.4
bparams.minThreshold = 100
bparams.maxThreshold = 250
bdet = cv2.SimpleBlobDetector_create(bparams)

firsttime = True
matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

seenmatches = []
matchstats = []
matchage = []
trackedhue = []
trackedsat = []

demorad = 256
democolors = ((255,255,0),(255,0,255),(0,255,255))
demoimg = np.zeros((demorad*2,demorad*2,3),np.uint8)
decay_factor = 50 # parameter for hue-saturation visualization

showcameraview = True
showdemoview = True

while(1):
    ret, img = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rev = 255 - gray
    keypoints = bdet.detect(rev)
    descriptors = np.array([point.pt for point in keypoints], np.float32)
    blank = np.zeros((1, 1))
    if not firsttime:
        matches = matcher.match(descriptors, lastdescriptors)
        for match in matches:
            lastloc = lastpoints[match.trainIdx].pt
            curloc = keypoints[match.queryIdx].pt
            currad = round(keypoints[match.queryIdx].size / 2)
            cx = round(curloc[1])
            cy = round(curloc[0])
            curpatch = img[cx-currad:cx+currad,cy-currad:cy+currad]
            hsvpatch = cv2.cvtColor(curpatch, cv2.COLOR_BGR2HSV)
            colavg = np.array([[np.mean(curpatch, axis=(0,1))]],dtype=np.uint8)
            hsvavg = cv2.cvtColor(colavg, cv2.COLOR_BGR2HSV)
            hue = float(hsvavg[0,0,0])
            saturation = float(hsvavg[0,0,1])
            try:
                matchidx = seenmatches.index(lastpoints[match.trainIdx])
            except:
                matchidx = -1
            if matchidx != -1:
                matchstats[matchidx] += 1
                seenmatches[matchidx] = keypoints[match.queryIdx]
                trackedhue[matchidx].append(hue)
                trackedsat[matchidx].append(saturation)
            else:
                seenmatches.append(keypoints[match.queryIdx])
                matchstats.append(2)
                matchage.append(2)
                trackedhue.append([hue])
                trackedsat.append([saturation])
        matchage = [age + 1 for age in matchage]
        for i in range(len(matchage))[::-1]:
            if matchage[i] > matchstats[i] + 1:
                matchstats.pop(i)
                seenmatches.pop(i)
                matchage.pop(i)
                trackedhue.pop(i)
                trackedsat.pop(i)
        print(len(trackedhue[0]))
        if showcameraview:
            blobs = cv2.drawKeypoints(img, seenmatches, blank, (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    elif showcameraview:
        blobs = cv2.drawKeypoints(img, keypoints, blank, (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    if showcameraview:
        cv2.imshow('image',blobs)
        if showdemoview:
            demoimg = cv2.subtract(demoimg, np.ones_like(demoimg) * decay_factor)
            for i in range(len(trackedhue)):
                dot = trackedhue[i]
                sat = trackedsat[i]
                if len(dot) > 5:
                    lastpoint = (demorad + demorad * (sat[-2] / 255) * cos(radians(dot[-2]*2)),
                        demorad + demorad * (sat[-2] / 255) * sin(radians(dot[-2]*2)))
                    curpoint = (demorad + demorad * (sat[-1] / 255) * cos(radians(dot[-1]*2)),
                        demorad + demorad * (sat[-1] / 255) * sin(radians(dot[-1]*2)))
                    lastpoint = np.int32(lastpoint)
                    curpoint = np.int32(curpoint)
                    demoimg = cv2.line(demoimg, lastpoint, curpoint, democolors[i%3], 10)
            cv2.imshow('demo',demoimg)
        x = cv2.waitKey(33)
        if x == 27:
            break
    lastpoints = keypoints
    lastdescriptors = descriptors
    firsttime = False

cap.release()
cv2.destroyAllWindows()

npcolors = [np.array(hue) for hue in trackedhue]

for dot in npcolors:
    plt.plot(dot)
plt.show()
