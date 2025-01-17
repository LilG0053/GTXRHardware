import cv2
import numpy as np
from matplotlib import pyplot as plt
from math import sin, cos, radians
import LEDMatch

# Video input from camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
cap.set(cv2.CAP_PROP_FPS, 60)
# cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,1)
# for i in range(10):
#     junk,junk2 = cap.read()
# cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,0)
cap.set(cv2.CAP_PROP_EXPOSURE, -9)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))

# Blob detection parameters
bparams = cv2.SimpleBlobDetector_Params()

# Filtering by area
bparams.filterByArea = True
bparams.minArea = 30
bparams.maxArea = 5000

# Filtering by circularity
bparams.filterByCircularity = False
bparams.minCircularity = 0.1

bparams.filterByConvexity = False

# Filtering by inertia
bparams.filterByInertia = False
bparams.minInertiaRatio = 0.1

# Filtering by brightness threshold
bparams.minThreshold = 50
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

showgraph = False

# Image loop
while(1):
    # Gets frame if available
    ret, img = cap.read()
    if not ret:
        break
    img = img[img.shape[0]//2 - 240:img.shape[0]//2 + 240,img.shape[1]//2 - 320:img.shape[1]//2 + 320]
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
        if len(descriptors) > 0 and len(lastdescriptors) > 0:
            matches = matcher.match(descriptors, lastdescriptors)
            for match in matches:
                # Gets previous and current locations of match
                curloc = keypoints[match.queryIdx].pt

                # Calculates hue and saturation of current match

                # Gets image area equivalent to square circumscribing blob
                # Checks to see if current match has been seen before (uses previous keypoint)
                matchedPoints = [match.getKeypoint() for match in seenmatches]
                matchidx = matchedPoints.index(lastpoints[match.trainIdx]) if lastpoints[match.trainIdx] in matchedPoints else -1
                if matchidx != -1:
                    currad = round(seenmatches[matchidx].getAvgSize() / 2)
                    # brightness = 127 * keypoints[match.queryIdx].size / seenmatches[matchidx].getAvgSize()
                else:
                    currad = round(keypoints[match.queryIdx].size / 2)
                    # brightness = 127
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
                # select brightness channel of single average pixel
                #brightness = float(hsvavg[0,0,2])
                if matchidx != -1:
                    brightness = float(hsvavg[0,0,2]) * keypoints[match.queryIdx].size / seenmatches[matchidx].getAvgSize()
                else:
                    currad = round(keypoints[match.queryIdx].size / 2)
                    brightness = float(hsvavg[0,0,2])

                # Checks to see if current match has been seen before (uses previous keypoint)
                # matchedPoints = [match.getKeypoint() for match in seenmatches]
                # matchidx = matchedPoints.index(lastpoints[match.trainIdx]) if lastpoints[match.trainIdx] in matchedPoints else -1
                if matchidx != -1:
                    # Updates match to use current point location
                    seenmatches[matchidx].update(keypoints[match.queryIdx], hue, saturation, brightness)
                    #print(f"Point: {matchidx} ID Sequence: {seenmatches[matchidx].getID()}")
                else:
                    # Adds new match to list
                    newMatch = LEDMatch.LEDMatch(keypoints[match.queryIdx], hue, saturation, brightness)
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
                brt = seenmatches[i].getBrightness()
                if len(dot) > 5:
                    # lastpoint = (demorad + demorad * (sat[-2] / 255) * cos(radians(dot[-2]*2)),
                    #     demorad + demorad * (sat[-2] / 255) * sin(radians(dot[-2]*2)))
                    # curpoint = (demorad + demorad * (sat[-1] / 255) * cos(radians(dot[-1]*2)),
                    #     demorad + demorad * (sat[-1] / 255) * sin(radians(dot[-1]*2)))
                    # lastpoint = (demorad * 2 * ((dot[-2]+25)%180) / 180,
                    #              demorad * 2 * brt[-2] / 255)
                    # curpoint = (demorad * 2 * ((dot[-1]+25)%180) / 180,
                    #              demorad * 2 * brt[-1] / 255)
                    lastpoint = (demorad + demorad * (brt[-2] / 255) * cos(radians(dot[-2]*2)),
                        demorad + demorad * (brt[-2] / 255) * sin(radians(dot[-2]*2)))
                    curpoint = (demorad + demorad * (brt[-1] / 255) * cos(radians(dot[-1]*2)),
                        demorad + demorad * (brt[-1] / 255) * sin(radians(dot[-1]*2)))
                    lastpoint = np.int32(lastpoint)
                    curpoint = np.int32(curpoint)
                    demoimg = cv2.line(demoimg, lastpoint, curpoint, democolors[i%3], 10)
            # for huerange in LEDMatch.hueranges:
            #     lowbound = np.int32([demorad + demorad * cos(radians(huerange[0]*2)),
            #         demorad + demorad * sin(radians(huerange[0]*2))])
            #     highbound = np.int32([demorad + demorad * cos(radians(huerange[1]*2)),
            #         demorad + demorad * sin(radians(huerange[1]*2))])
            #     center = (demorad, demorad)
            #     demoimg = cv2.line(demoimg, center, lowbound, (0,0,255), 3)
            #     demoimg = cv2.line(demoimg, center, highbound, (0,255,0), 3)
            cv2.imshow('demo',demoimg)
        x = cv2.waitKey(1)
        if x == 27:
            break
        if x & 0xFF == ord('u'):
            cap.set(cv2.CAP_PROP_EXPOSURE, cap.get(cv2.CAP_PROP_EXPOSURE)+1)
        if x & 0xFF == ord('d'):
            cap.set(cv2.CAP_PROP_EXPOSURE, cap.get(cv2.CAP_PROP_EXPOSURE)-1)

    # Shifts keypoint and descriptor references
    lastpoints = keypoints
    lastdescriptors = descriptors

    
    firsttime = False

# Cleanup code
cap.release()
cv2.destroyAllWindows()

if showgraph:
    npcolors = [np.array(match.getHue()) for match in seenmatches]

    for dot in npcolors:
        plt.plot(dot)
    plt.show()
