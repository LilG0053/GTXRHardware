import cv2
import numpy as np
from matplotlib import pyplot as plt
from math import sin, cos, radians

from LEDMatch import LEDMatch

# Video input from file
cap = cv2.VideoCapture('output2.avi')

# Blob detection parameters
bparams = cv2.SimpleBlobDetector_Params()

# Filtering by area
bparams.filterByArea = True
bparams.minArea = 30
bparams.maxArea = 1000

# Filtering by circularity
bparams.filterByCircularity = False
bparams.minCircularity = 0.2

bparams.filterByConvexity = False

# Filtering by inertia
bparams.filterByInertia = True
bparams.minInertiaRatio = 0.1

# Filtering by brightness threshold
bparams.minThreshold = 100
bparams.maxThreshold = 255

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
        # Finds matches of current keypoint locations with previous keypoint locations
        matches = matcher.match(descriptors, lastdescriptors)
        for match in matches:
            # Gets previous and current locations of match
            curloc = keypoints[match.queryIdx].pt

            # Calculates hue and saturation of current match

            # Gets image area equivalent to square circumscribing blob
            currad = round(keypoints[match.queryIdx].size / 2)
            cx = round(curloc[1])
            cy = round(curloc[0])
            curpatch = img[cx-currad:cx+currad,cy-currad:cy+currad]
            # find average RGB color
            colavg = np.array([[np.mean(curpatch, axis=(0,1))]],dtype=np.uint8)
            # convert average RGB color to HSV
            hsvavg = cv2.cvtColor(colavg, cv2.COLOR_BGR2HSV)
            # select hue channel of single average pixel
            hue = float(hsvavg[0,0,0])
            # select saturation channel of single average pixel
            saturation = float(hsvavg[0,0,1])
            # gets lightness channel of average pixel
            lightness = float(hsvavg[0,0,2])

            # Checks to see if current match has been seen before (uses previous keypoint)
            matchedPoints = [match.getKeypoint() for match in seenmatches]
            matchidx = matchedPoints.index(lastpoints[match.trainIdx]) if lastpoints[match.trainIdx] in matchedPoints else -1
            if matchidx != -1:
                # Updates match to use current point location
                seenmatches[matchidx].update(keypoints[match.queryIdx], hue, saturation, lightness)
                print(f"Point: {matchidx} ID Sequence: {seenmatches[matchidx].getID()}")
            else:
                # Adds new match to list
                newMatch = LEDMatch(keypoints[match.queryIdx], hue, saturation, lightness)
                seenmatches.append(newMatch)

        # Removes every match where the age is too old
        for i in range(len(seenmatches))[::-1]:
            if not seenmatches[i].checkAge():
                seenmatches.pop(i)

        # Draws blobs
        if showcameraview:
            matchedPoints = [match.getKeypoint() for match in seenmatches]
            blobs = cv2.drawKeypoints(img, matchedPoints, blank, (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    elif showcameraview:
        # Draws blobs
        blobs = cv2.drawKeypoints(img, keypoints, blank, (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    if showcameraview:
        for match in seenmatches:
            if match.getID().size >= 3:
                matchindex = -1
                for i in range(len(match.seqlist)):
                    if np.array_equal(match.seqlist[i],match.getID()):
                        matchindex = i
                        break
                if matchindex != -1:
                    bgrcolor = cv2.cvtColor(np.array([[[180*(matchindex/len(match.seqlist)),255,255]]],dtype=np.uint8),cv2.COLOR_HSV2BGR)[0][0].tolist()
                    blobs = cv2.circle(blobs,np.int32(match.getKeypoint().pt),10,bgrcolor,-1) #(255,0,0)
        cv2.imshow('image',blobs)
        # Shows points on hue saturation plot
        if showdemoview:
            demoimg = cv2.subtract(demoimg, np.ones_like(demoimg) * decay_factor)
            for i in range(len(seenmatches)):
                dot = seenmatches[i].getHue()
                sat = seenmatches[i].getSaturation()
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

npcolors = [np.array(match.getHue()) for match in seenmatches]

for dot in npcolors:
    plt.plot(dot)
plt.show()
