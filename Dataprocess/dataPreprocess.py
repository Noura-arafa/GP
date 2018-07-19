import numpy as np
from PIL import Image

class dataPreprocess:
    types = str, int, int

    def readingDatasetfromfile(self,filename):
        with open(filename, "r") as inputfile:
            data = [tuple(t(e) for t, e in zip(self.types, line.split()))
                    for line in inputfile.readlines()]
        return data

    def paddingbatchclips(allClips, longestLen):
        fixedClipsize = []
        for clip in allClips:
            clip = np.pad(clip, (0, longestLen - len(clip)), 'constant', constant_values=(0))
            fixedClipsize.append(clip)
        return fixedClipsize

    def readingClipsarrayfromfile(clipsfilename):
        with open(clipsfilename, 'r') as fo:
            for line in fo:
                fields = line.split(',')
                smallarray = []
                for item in fields:
                    smallarray.append(np.float64(item))
        return smallarray