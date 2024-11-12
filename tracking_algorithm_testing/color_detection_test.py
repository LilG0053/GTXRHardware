import cv2
import numpy as np
from matplotlib import pyplot as plt
from math import sin, cos, radians


class LEDMatch:
    ageThres = 2
    # Hue ranges for identifying colors
    hueranges = np.array([[160, 20], [75, 100], [100, 125]])
    # Color for synchronization pulses
    syncColor = 0
    # Sequence length
    seqLength = 2
    def __init__(self, keypoint, hue, saturation):
        self.hue = []
        self.saturation = []

        self.id = np.array([])

        # Whether in the middle of a sync pulse
        self.onSyncColor = False

        # Whether code is being transmitted
        self.syncing = False

        # How many sync pulse rising edges
        self.syncNum = 0

        # Buffer for color sequence
        self.buffer = []

        self.update(keypoint, hue, saturation)
        pass
    def getKeypoint(self):
        return self.keypoint
    def getHue(self):
        return self.hue
    def getSaturation(self):
        return self.saturation
    def getID(self):
        return self.id
    def update(self, keypoint, hue, saturation):
        self.keypoint = keypoint
        self.hue.append(hue)
        self.saturation.append(saturation)
        self.age = 0

        # Checks detected hue against possible ranges
        for i in range(len(self.hueranges)):
            detected = -1
            if self.hueranges[i, 0] < self.hueranges[i, 1]:
                # If range does not wrap around
                if hue > self.hueranges[i, 0] and hue < self.hueranges[i, 1]:
                    detected = i
            else:
                # If range does wrap around
                if hue > self.hueranges[i, 0] or hue < self.hueranges[i, 1]:
                    detected = i
        
        if detected == self.syncColor and (not self.onSyncColor):
            # If rising edge of sync pulse
            self.onSyncColor = True

            # Increments rising edge count
            self.syncNum+=1

            # Turns on syncing mode and resets buffer
            if not self.syncing:
                self.syncing = True
                self.buffer = []
        if detected != self.syncColor and self.onSyncColor:
            # If falling edge of sync pulse
            self.onSyncColor = False
            if self.syncing and self.syncNum >= 2:
                # If falling edge is after second sync pulse, code is completed
                self.syncing = False
                self.syncNum = 0

                code = np.array(self.buffer, dtype=np.int16)

                # Detects falling and rising edges of sync pulse
                syncLoc = np.diff((code == self.syncColor).astype(np.int16))

                # Gets center of the first and second sync pulses
                syncStart = np.where(syncLoc == -1)[0][0]/2.
                syncEnd = (np.where(syncLoc == 1)[0][0]+len(syncLoc)+1)/2.

                # Gets intervals where the individual colors should be based on expected sequence length (note: includes sync pulses)
                # This uses a method like a bar code to figure out where the bars are by making a sort of ruler between calibration bars
                intervals = np.linspace(syncStart, syncEnd, self.seqLength + 2)

                # Gets the actual sequence using the intervals not including the sync pulses
                sequence = code[np.round(intervals[1:len(self.hueranges)]).astype(np.int16)]
                self.id = sequence

        if self.syncing:
            # Adds to buffer if recording sequence
            self.buffer.append(detected)
    def checkAge(self):
        self.age += 1
        return self.age < self.ageThres

# Video input from file
cap = cv2.VideoCapture('output2.avi')

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

            # Checks to see if current match has been seen before (uses previous keypoint)
            matchedPoints = [match.getKeypoint() for match in seenmatches]
            matchidx = matchedPoints.index(lastpoints[match.trainIdx]) if lastpoints[match.trainIdx] in matchedPoints else -1
            if matchidx != -1:
                # Updates match to use current point location
                seenmatches[matchidx].update(keypoints[match.queryIdx], hue, saturation)
                print(f"Point: {matchidx} ID Sequence: {seenmatches[matchidx].getID()}")
            else:
                # Adds new match to list
                newMatch = LEDMatch(keypoints[match.queryIdx], hue, saturation)
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
        # Shows points on hue saturation plot
        cv2.imshow('image',blobs)
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
