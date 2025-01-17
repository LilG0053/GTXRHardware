import numpy as np

class LEDMatch:
    ageThres = 2
    # Hue ranges for identifying colors
    hueranges = np.array([[160, 20], [25, 65], [70, 110], [115, 155]])
    # Color for synchronization pulses
    offColor = 0
    # Sequence length
    seqLength = 3
    # number of zero colors until packet end is assumed
    blankThreshold = 10
    # Valid color sequence list
    seqlist = [[1,2,1],[1,2,2],[1,2,3],[2,1,3],[2,1,2],[2,3,1],[3,1,3],[3,1,2],[3,2,3]]
    # seqlist = [ [1,0,1,0,1,1,0,1], [1,0,1,1,0,1,0,1], [1,0,0,1,0,1,0,1], [1,0,0,1,1,0,0,1], [1,0,0,0,1,1,0,1], [1,0,0,0,1,0,0,1], [1,0,1,1,1,0,0,1], [1,0,1,0,0,1,0,1], [1,0,0,1,1,1,0,1] ]
    seqlist = [np.array(element) for element in seqlist]
    def __init__(self, keypoint, hue, saturation, brightness):
        self.hue = []
        self.saturation = []
        self.brightness = []

        self.id = np.array([])

        # average size of marker (for stability + signaling)
        self.avgSize = keypoint.size

        # Whether code is being transmitted
        self.syncing = False

        # How many sync pulse rising edges
        self.syncNum = 0

        # number of blank colors
        self.numBlanks = 0

        # Buffer for color sequence
        self.buffer = []

        self.update(keypoint, hue, saturation, brightness)
        pass
    def getKeypoint(self):
        return self.keypoint
    def getHue(self):
        return self.hue
    def getSaturation(self):
        return self.saturation
    def getBrightness(self):
        return self.brightness
    def getAvgSize(self):
        return self.avgSize
    def getID(self):
        return self.id
    def update(self, keypoint, hue, saturation, brightness):
        #print("Updating with  hue %d:"%(hue))
        self.keypoint = keypoint
        self.hue.append(hue)
        self.saturation.append(saturation)
        self.brightness.append(brightness)
        self.age = 0
        self.avgSize = self.avgSize * 0.96 + keypoint.size * 0.04

        # Checks detected hue against possible ranges
        detected = -1
        for i in range(len(self.hueranges)):
            if self.hueranges[i, 0] < self.hueranges[i, 1]:
                # If range does not wrap around
                if hue > self.hueranges[i, 0] and hue < self.hueranges[i, 1]:
                    detected = i
            else:
                # If range does wrap around
                if hue > self.hueranges[i, 0] or hue < self.hueranges[i, 1]:
                    detected = i
        if detected != -1:
            if detected != self.offColor:
                if not self.syncing:
                    self.syncing = True
                    self.buffer = []
                self.syncNum += 1
            else:
                if self.syncing:
                    self.syncing = False
                    self.syncNum = 0

                    code = np.array(self.buffer, dtype=np.int16)
                    print("buffer: " + str(code))

                    # Gets intervals where the individual colors should be based on expected sequence length (note: includes sync pulses)
                    # This uses a method like a bar code to figure out where the bars are by making a sort of ruler between calibration bars
                    startind = ((len(code) / self.seqLength) - 1) / 2
                    intervals = np.linspace(startind, len(code) - 1 - startind, self.seqLength)
                    # print("intervals: "+str(intervals))

                    # Gets the actual sequence using the intervals not including the sync pulses
                    sequence = code[np.round(intervals).astype(np.int16)]
                    # print("sequence: "+str(sequence))
                    self.id = sequence
            if self.syncing:
                # Adds to buffer if recording sequence
                self.buffer.append(detected)
                if len(self.buffer) > 40:
                    self.buffer = self.buffer[-40:]
    def checkAge(self):
        self.age += 1
        return self.age < self.ageThres