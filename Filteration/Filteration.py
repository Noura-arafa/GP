import moviepy.editor as mp
import cv2
from mutagen.mp3 import MP3
from moviepy.editor import *
from Model.LSTMmodel import lstmModel
import itertools
import numpy as np
import matplotlib.image as mpimg
import scipy.misc
import imageio
imageio.plugins.ffmpeg.download()


# # #-----------to get audio from clip----------
class Filteration:
    filteredClasses = [1]
    LSTMModel = lstmModel()
    clipsFrames = []
    numberofFrames = 90

    def seperateAduioFromVideo(self, videoName, videoPath):
        clip = mp.VideoFileClip(videoPath + videoName + ".mp4")
        clip.audio.write_audiofile(videoName + ".mp3")

    def cutIntoClips(self, videoPath, videoName):
        my_clip = VideoFileClip(videoPath + videoName + ".mp4")
        length = my_clip.duration
        Clipscounter = 0
        i = 0
        while i < int(length):
            subclip = my_clip.subclip(i, i + 3)
            subclip.write_videofile(videoPath+"\\"+str(Clipscounter) + ".mp4")
            subclip.audio.write_audiofile(videoPath+"\\"+str(Clipscounter) + ".mp3")
            i += 3
            Clipscounter += 1
        return Clipscounter

    def cutIntoFrames(self, clipPath, videoname, clipnumber):
        cap = cv2.VideoCapture(clipPath + "\\" + str(clipnumber) + ".mp4")
        success, image = cap.read()
        currentFrame = 0
        newpath = clipPath + "\\" + str(clipnumber)
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        os.chdir(newpath)
        while (success):
            os.chdir(newpath)
            # Capture frame-by-frame
            success, frame = cap.read()

            # Saves image of the current frame in jpg file

            name = 'frame' + str(currentFrame) + '.jpg'
            cv2.imwrite(name, frame)

            # To stop duplicate images
            currentFrame += 1

    def rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    def readframesFromfile(self, filename, num_frames):
        i = 0;
        clip = []
        Img = []
        clipframe = []
        for i in range(num_frames):
            framenumber = "frame" + str(i) + ".jpg"
            img = mpimg.imread(filename + "\\" + framenumber)
            clipframe.append(img)

            # gray is a numpy array
            gray = self.rgb2gray(img)
            img = cv2.resize(gray, (100, 100))
            img = np.hstack(img)

            clip.append(img)
            # list of 1D arrays(frames)

        return clip, clipframe

    def createClip(self, clipPath):
        labels = []
        images = [img for img in os.listdir(clipPath+"\\") if img.endswith(".jpg")]
        Clip, clipframe = self.readframesFromfile(clipPath, len(images)-2)
        Clip = list(itertools.chain.from_iterable(Clip))
        return Clip, clipframe

    def getPrediction(self, Clips):
        model_path = "E:\FOURTH YEAR\GP\youTube control\\2_class_sessions\epoch 14"
        prediction = self.LSTMModel.filter(Clips, model_path)
        return prediction

    def Filtervideo(self, videoname, videoPath):
        Clips = []
        clipscounter = self.cutIntoClips(videoPath, videoname)

        for clip in range(clipscounter):
            self.cutIntoFrames(videoPath, videoname, clip)
        for clipcount in range(clipscounter):
            clip, clipframe = self.createClip(videoPath + "\\" +str(clipcount)+"\\" )
            Clips.append(clip)
            self.clipsFrames.append(clipframe)
        Prediction = self.getPrediction(Clips)
        outputClip = []
        numOfBlurframes = 0

        thefile = open(videoPath+"\\"+"out.txt", 'w')

        for i in range(len(Prediction)):
            itemindex = np.where(Prediction[i] == max(Prediction[i]))
            if (itemindex[0][0] in self.filteredClasses):

                numOfBlurframes += 1
                os.chdir(videoPath)
                fmpeg_command = 'ffmpeg -i '+str(i)+'.mp4 -vf "boxblur=50:10" blur'+str(i)+'.mp4'
                os.system(fmpeg_command)
                x="'"
                thefile.write("%s " %"file")
                thefile.write("%s" %x)
                thefile.write("%s" %"blur"+str(i)+".mp4"+x)
                thefile.write("\n")
            else:
                x="'"
                thefile.write("%s  " %"file")
                thefile.write("%s" %x)
                thefile.write("%s" %str(i)+".mp4"+x)
                thefile.write("\n")


            thefile = open(videoPath+"\\"+"out.txt", 'a')

        #if (len(Prediction) / 2 > numOfBlurframes):
        os.chdir(videoPath)
        ffmpeg_command1 = 'ffmpeg -f concat -i out.txt -c copy output.mp4'
        os.system(ffmpeg_command1)

    def bluring(self, clip, clipPath):
        for i in range(len(clip)):
            blurImg = cv2.blur(clip[i], (200, 250))
            scipy.misc.imsave(clipPath + "\\" + "frame" + str(i) + ".jpg", blurImg)


    # --------get duration of the audio------------
    def get_audio_duration(self, audio_name):
        audio = MP3(audio_name)
        return audio.info.length

    # ----------------Make video by Frames----------------------------------------------
    def make_video(self, videoPath, frames_folder, video_name):
        # ----read all frames name from frames folder--
        images = [img for img in os.listdir(frames_folder) if img.endswith(".jpg")]
        frame = cv2.imread(os.path.join(frames_folder, images[0]))
        height, width, layers = frame.shape
        number_frames = len(images)
        audio_duration = self.get_audio_duration(videoPath + video_name + ".mp3")
        fps = number_frames-2 / audio_duration
        video = cv2.VideoWriter(videoPath + "blur" + video_name + ".mp4", cv2.VideoWriter_fourcc('F', 'M', 'P', '4'),
                                fps, (width, height))

        for i in range(number_frames-2):
            framenumber = "frame" + str(i) + ".jpg"
            video.write(cv2.imread(os.path.join(frames_folder, framenumber)))
        cv2.destroyAllWindows()

    # ---------------------put audio on video------------------------
    def merge_audio_video(self, video_name,video):
        video1 = mp.VideoFileClip(video_name)
        audio_name = video + ".mp3"
        output_video_name = video_name
        return video1.write_videofile(output_video_name, audio=audio_name)


filter = Filteration()
testcasename="TestCase2"
testcasepath= "E:\FOURTH YEAR\GP\youTube control\Filteration\\"+testcasename+"\\"
filter.Filtervideo(testcasename, testcasepath)