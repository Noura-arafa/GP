import tensorflow as tf
from tensorflow.contrib import rnn
import keras
from keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import cv2
import itertools
from os import listdir
from os.path import isfile, join
import matplotlib.image as mpimg

types = str, int, int
#
# bad_labels = []
# with open("labels.txt") as f:
#     bad_labels = [int(x) for x in f.read().split()]


def readingDatasetfromfile(filename):
    with open(filename, "r") as inputfile:
        data = [tuple(t(e) for t, e in zip(types, line.split()))
                for line in inputfile.readlines()]
    return data


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def readframesFromfile(filename, num_frames):
    i = 0;
    clip = []
    Img = []

    for i in range(num_frames):
        framenumber = "frame" + str(i) + ".jpg"
        img = mpimg.imread(filename + "\\" + framenumber)
        # gray is a numpy array
        gray = rgb2gray(img)
        img = cv2.resize(gray, (50, 50))
        img = np.hstack(img)

        clip.append(img)
        # list of 1D arrays(frames)
    return clip


def createClips(filename):
    data = readingDatasetfromfile(filename)
    Clips = []
    labels = []
    maximumnumberframes = 0
    for x in data:
        num_frames = [f for f in listdir("E:\\FourthYear\\3-GP\\GP-Git\\Clips\\Frames\\" + x[0] + "\\" + x[0] + "," + str(x[1])) if
                           isfile(join("E:\\FourthYear\\3-GP\\GP-Git\\Clips\\Frames\\" + x[0] + "\\" + x[0] + "," + str(x[1]), f))]
        maximumnumberframes = max(len(num_frames)-2, maximumnumberframes)

        Clip = readframesFromfile("E:\\FourthYear\\3-GP\\GP-Git\\Clips\\Frames\\" + x[0] + "\\" + x[0] + "," + str(x[1]) + "\\", len(num_frames)-2)
        print("film name", x[0])
        Clip = list(itertools.chain.from_iterable(Clip))
        Clips.append(Clip)
        labels.append(x[2])
    return Clips, labels, maximumnumberframes


def paddingbatchclips(allClips, longestLen):
    print(longestLen)
    fixedClipsize = []
    for clip in allClips:
        clip = np.pad(clip, (0, longestLen - len(clip)), 'constant', constant_values=(0))
        fixedClipsize.append(clip)
        #to get the numberofframes we devide the clipsize / framesize
        #  this // to get an intger value instead of using / that return float value
    return fixedClipsize


def make_cell(lstm_size):
  return tf.nn.rnn_cell.BasicLSTMCell(lstm_size, state_is_tuple=True)
# lstm_layer = rnn.BasicLSTMCell(num_units, forget_bias=1)


Clips, labels, maxnumberframes = createClips("TestData.txt")
Clips = paddingbatchclips(Clips, maxnumberframes*50*50)
print(maxnumberframes)
#MODEL
#attributes
learning_rate = 0.001
n_classes = 80
num_units = 128
# frame size w*h
n_input = 2500
#weights and bias
weights = tf.Variable(tf.random_normal([num_units, n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))

#placeholders
x = tf.placeholder("float", [None, None, n_input])
y = tf.placeholder("float", [None, n_classes])
# numberframes = tf.placeholder(tf.int32)


#---number of lstm layers---
numlayer = 3


#-------create 3 layer of lstm cell and return prediction------
def LSTMmodel(batch_x, numberOfframes, weight, Bias):
    input = tf.unstack(batch_x, numberOfframes, 1)
    cell = tf.nn.rnn_cell.MultiRNNCell([make_cell(num_units) for _ in range(numlayer)], state_is_tuple=True)
    outputs, _ = rnn.static_rnn(cell, input, dtype="float32")
    # converting last output of dimension [batch_size,num_units] to [batch_size,n_classes] by out_weight multiplication
    # (output*weight)+bias
    prediction = tf.matmul(outputs[-1], weight) + Bias
    return prediction


pred = LSTMmodel(x, maxnumberframes, weights, bias)
# loss_function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
# optimization
opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# model evaluation
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


epoch = 0
#15
epochs = 1
#10000
trainDataSize = 1
#50
batchsize = 1
#50
batchlimit = 1
dataIterator = 0
batchIterator = 0
#---folder path to save sess of the model in it----
model_path = "./model"

init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    while epoch < epochs:
        while dataIterator < trainDataSize:
             batch_x = Clips[batchIterator:batchlimit]
             actions = tf.one_hot(labels[batchIterator:batchlimit], depth=80, on_value=1.0, off_value=0.0, axis=1)
             batch_y = tf.Session().run(actions)
             # batch_x, number_Frames = paddingbatchclips(batch_x)
             batch_x = np.array([np.array(xi) for xi in batch_x])
             batch_x = batch_x.reshape((batchsize, maxnumberframes, n_input))
             batch_y = np.array([np.array(yi) for yi in batch_y])
             batchIterator = batchlimit
             batchlimit += batchsize
             print("dataIterator ", dataIterator)
             oprimization = sess.run(opt, feed_dict={x: batch_x, y: batch_y})
             acc=sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
             los=sess.run(loss, feed_dict={x: batch_x, y: batch_y})
             pred = sess.run(pred, feed_dict={x: batch_x, y: batch_y})
             dataIterator += batchsize
             print("Accurecy : ", acc)
             print("prediction : ", pred)
        epoch += 1
    save_path=saver.save(sess, model_path)

