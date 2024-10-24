import cv2
import numpy as np
from matplotlib import pyplot as plt
from math import sin, cos, radians

# Video input from file
cap = cv2.VideoCapture('output.avi')

# Blob detection parameters
bparams = cv2.SimpleBlobDetector_Params()

# Filtering by area
bparams.filterByArea = True
bparams.minArea = 30
bparams.maxArea = 300

# Filtering by circularity
bparams.filterByCircularity = True
bparams.minCircularity = 0.5

bparams.filterByConvexity = False

# Filtering by inertia
bparams.filterByInertia = True
bparams.minInertiaRatio = 0.4

# Filtering by brightness threshold
bparams.minThreshold = 100
bparams.maxThreshold = 250

# Blob detection instance from params
bdet = cv2.SimpleBlobDetector_create(bparams)

# Whether it is the first loop iteration
firsttime = True

# Point matcher
matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Current keypoints
keypoints = []
# Previous keypoints
lastpoints = []


# Current matrix of keypoint locations
descriptors = np.array([])
# Previous matrix of keypoint locations
lastdescriptors = np.array([])

# List of matched points, which is updated with the most recent location of each point
seenmatches = []
# How many times each point has been successfully matched
matchstats = []
# How many frames each match has existed for
matchage = []
# The hue of each matched point
trackedhue = []
# The saturation of each matched point
trackedsat = []

demorad = 256
democolors = ((255,255,0),(255,0,255),(0,255,255))
demoimg = np.zeros((demorad*2,demorad*2,3),np.uint8)
decay_factor = 50 # parameter for hue-saturation visualization

# Whether blob detection is displayed
showcameraview = True

# Whether tracked point hue/saturation view is displayed
showdemoview = True

# Image loop
while(1):
    # Gets frame if available
    ret, img = cap.read()
    if not ret:
        break
    # Converts to inverted grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rev = 255 - gray

    # Gets keypoints from blob detector
    keypoints = bdet.detect(rev)

    # Gets matrix of keypoint locations
    descriptors = np.array([point.pt for point in keypoints], np.float32)
    blank = np.zeros((1, 1))
    if not firsttime:
        # Finds matches of current keypoint locations with previous kepoint locations
        matches = matcher.match(descriptors, lastdescriptors)
        for match in matches:
            # Gets previous and current locations of match
            lastloc = lastpoints[match.trainIdx].pt
            curloc = keypoints[match.queryIdx].pt

            # Calculates hue and saturation of current match

            # Gets image area equivalent to square circumscribing blob
            currad = round(keypoints[match.queryIdx].size / 2)
            cx = round(curloc[1])
            cy = round(curloc[0])
            curpatch = img[cx-currad:cx+currad,cy-currad:cy+currad]
            # Converts to HSV
            hsvpatch = cv2.cvtColor(curpatch, cv2.COLOR_BGR2HSV)
            # No idea what this does
            colavg = np.array([[np.mean(curpatch, axis=(0,1))]],dtype=np.uint8)
            hsvavg = cv2.cvtColor(colavg, cv2.COLOR_BGR2HSV)
            # Somehow comes out with average hue and saturation?
            hue = float(hsvavg[0,0,0])
            saturation = float(hsvavg[0,0,1])

            # Checks to see if current match has been seen before (uses previous keypoint)
            matchidx = seenmatches.index(lastpoints[match.trainIdx])
            if matchidx != -1:
                # Increments number of times point has been successfully matched
                matchstats[matchidx] += 1
                # Updates match to use current point location
                seenmatches[matchidx] = keypoints[match.queryIdx]

                # Updates hue and saturation of matched point
                trackedhue[matchidx].append(hue)
                trackedsat[matchidx].append(saturation)
            else:
                # Adds new match to list
                seenmatches.append(keypoints[match.queryIdx])
                # Automatically has been seen twice and has age of 2
                matchstats.append(2)
                matchage.append(2)
                # Adds hue and saturation
                trackedhue.append([hue])
                trackedsat.append([saturation])
        # Increments ages of everything
        matchage = [age + 1 for age in matchage]
        # Removes every match where the age is at least 2 ahead of the count
        for i in range(len(matchage))[::-1]:
            if matchage[i] > matchstats[i] + 1:
                matchstats.pop(i)
                seenmatches.pop(i)
                matchage.pop(i)
                trackedhue.pop(i)
                trackedsat.pop(i)
        print(len(trackedhue[0]))
        # Draws blobs
        if showcameraview:
            blobs = cv2.drawKeypoints(img, seenmatches, blank, (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    elif showcameraview:
        # Draws blobs
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

    # Shifts keypoint and descriptor references
    lastpoints = keypoints
    lastdescriptors = descriptors

    
    firsttime = False

# Cleanup code
cap.release()
cv2.destroyAllWindows()

npcolors = [np.array(hue) for hue in trackedhue]

for dot in npcolors:
    plt.plot(dot)
plt.show()
