import cv2
import numpy as np
import os

types = str, int
#al input 1 dah al folder ely feh al clips we odmha al tipme stamp
with open("input1.txt", "r") as inputfile:
    data = [tuple(t(e) for t,e in zip(types, line.split()))
                for line in inputfile]
print(data)
clip_FolderName = "E:\FourthYear\\3-GP\\GP-Git\\Clips\\Clips\\"
frames_FolderName = "E:\FourthYear\\3-GP\\GP-Git\\Clips\\Frames\\"

for x in data:
    #l7ad cutting dah al mkan ely feh al folder ely gwa al clips
    cap = cv2.VideoCapture(clip_FolderName+x[0]+"\\"+x[0]+","+str(x[1])+".mp4")
    success, image = cap.read()
    currentFrame = 0
    newpath = frames_FolderName + x[0] + "\\" + x[0] + "," + str(x[1])
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    os.chdir(newpath)
    while (success):
        os.chdir(newpath)
        # Capture frame-by-frame
        success, frame = cap.read()

        # Saves image of the current frame in jpg file

        name ='frame' + str(currentFrame) + '.jpg'
        print('Creating...' + name)
        cv2.imwrite(name, frame)

        # To stop duplicate images
        currentFrame += 1

    # cap.release()
    # cv2.destroyAllWindows()



