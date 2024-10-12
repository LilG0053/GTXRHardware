import cv2
import numpy as np
from matplotlib import pyplot as plt

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

sift = cv2.SIFT_create()
seenmatches = []
matchstats = []
matchage = []
trackedcolors = []

showcameraview = False

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
        goodpts = []
        for match in matches:
            lastloc = lastpoints[match.trainIdx].pt
            curloc = keypoints[match.queryIdx].pt
            currad = round(keypoints[match.queryIdx].size / 2)
            cx = round(curloc[1])
            cy = round(curloc[0])
            curpatch = img[cx-currad:cx+currad,cy-currad:cy+currad]
            colavg = np.mean(curpatch, axis=(0,1))
            hue = float(colavg[0])
            try:
                matchidx = seenmatches.index(lastpoints[match.trainIdx])
            except:
                matchidx = -1
            if matchidx != -1:
                matchstats[matchidx] += 1
                seenmatches[matchidx] = keypoints[match.queryIdx]
                trackedcolors[matchidx].append(hue)
            else:
                seenmatches.append(keypoints[match.queryIdx])
                matchstats.append(2)
                matchage.append(2)
                trackedcolors.append([hue])
            goodpts.append(keypoints[match.queryIdx])
        matchage = [age + 1 for age in matchage]
        for i in range(len(matchage))[::-1]:
            if matchage[i] > matchstats[i] + 1:
                matchstats.pop(i)
                seenmatches.pop(i)
                matchage.pop(i)
                trackedcolors.pop(i)
        print(len(trackedcolors[0]))
        if showcameraview:
            blobs = cv2.drawKeypoints(img, seenmatches, blank, (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    elif showcameraview:
        blobs = cv2.drawKeypoints(img, keypoints, blank, (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    if showcameraview:
        cv2.imshow('image',blobs)
        x = cv2.waitKey(33)
        if x == 27:
            break
    lastpoints = keypoints
    lastdescriptors = descriptors
    firsttime = False

cap.release()
cv2.destroyAllWindows()

npcolors = np.array(trackedcolors)

for dot in npcolors:
    plt.plot(dot)
plt.show()