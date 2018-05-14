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

types = str, int, int
with open("data.txt", "r") as inputfile:
    data = [tuple(t(e) for t,e in zip(types, line.split()))
                for line in inputfile.readlines()]
bad_labels = []
with open("labels.txt") as f:
    bad_labels = [int(x) for x in f.read().split()]


def readImagesFromfile(filename, num_frames):
    i=0;
    inputImages =[]
    Img=[]
    for i in range(num_frames):
        framenumber="frame"+str(i)+".jpg"
        im = Image.open(filename+framenumber).convert('LA')
        im.save('greyscale'+str(i) + '.png')
        im=cv2.imread('greyscale'+str(i)+'.png', 0)
        im = cv2.resize(im, (100, 100))
        z = np.hstack(im)
        inputImages.append(z)
        #list of 1Darrays(frames)
    return inputImages


def LSTMmodel(number_Frames,batch_x,batch_y):
    learning_rate = 0.001
    n_classes = 1
    num_units = 128
    weights = tf.Variable(tf.random_normal([num_units, n_classes]))
    bias = tf.Variable(tf.random_normal([n_classes]))
    x =tf.placeholder("float",[None,None,n_input])
    y = tf.placeholder("float",[None,1])
    # [batchsize,n_input]
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)
    input = tf.unstack(x, number_Frames, 1)
    '''
    The Network
    '''
    lstm_layer = rnn.BasicLSTMCell(num_units, forget_bias=1)
    outputs, _ = rnn.static_rnn(lstm_layer, input, dtype="float32")
    # converting last output of dimension [batch_size,num_units] to [batch_size,n_classes] by out_weight multiplication
    # (output*weight)+bias
    prediction = tf.matmul(outputs[-1], weights) + bias
    # loss_function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    # optimization
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    # model evaluation
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        sess.run(opt, feed_dict={x: batch_x, y: batch_y})
        acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
        los = sess.run(loss, feed_dict={x: batch_x, y: batch_y})
    return correct_prediction,acc,los

def clipsFixedsize(allClips):
    #print(len(allClips))
    fixedClipsize=[]
    longestLen=len(max(allClips,key=len))
    for clip in allClips:
        if(len(clip)<longestLen):
            diff=longestLen-len(clip)
            iter=0
            while iter < diff :
                clip.append(0)
                iter+=1
            fixedClipsize.append(clip)
        else:
            fixedClipsize.append(clip)
    return fixedClipsize,longestLen


n_input = 10000
batchsize = 2
Clips=[]
labels= []
#read clip frames and then convert it to 1d array and put it in clips array
#get num-frames in the specified folder
for x in data:
    num_frames = [f for f in listdir("Clips\\Frames\\"+x[0]+","+str(x[1])) if
                 isfile(join("Clips\\Frames\\"+x[0]+","+str(x[1]), f))]
    Clip=readImagesFromfile("Clips\\Frames\\"+x[0]+","+str(x[1]), num_frames)
    Clip = list(itertools.chain.from_iterable(Clip1))
    Clips.append(Clip)
    labels.append(x[1])

init= tf.global_variables_initializer()
iter1 = 0
iter2 = 0
while iter1<1:
    batch_x=[]
    _x= []
    _label= []
    while iter2<batchsize:
       # print('image size ', len(ImagesPixelsarray), ' iter2 ' , iter2)
        #print("the Clip",Clips[iter2])
        _x.append(Clips[iter2])
        _label.append(labels[iter2])
    iter2+=1
    batchsize+=1
    print("X",_x)
    #make number of frames of that batch fixed
    _x, number_Frames = clipsFixedsize(_x)
    #convert x and y to numpy array so it will work with lstm
    batch_x= np.array([np.array(xi) for xi in _x])
    batch_y= np.array([np.array(yi) for yi in _label])
    print('batch numpy array',batch_x)
    batch_x = batch_x.reshape((batchsize-1, number_Frames, n_input))
    correct_prediction,acc,los = LSTMmodel(number_Frames, batch_x, batch_y)
    if iter1 %2==0:
         print("For iter ", iter1)
         print("Accuracy ", acc)
         print("Loss ", los)

    iter1+=1