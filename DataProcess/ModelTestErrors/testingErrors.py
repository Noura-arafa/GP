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










bad_labels = []
with open("labels.txt") as f:
    bad_labels = [int(x) for x in f.read().split()]

def readingDatafromfile(filename):
    with open(filename, "r") as inputfile:
        data = [tuple(t(e) for t, e in zip(types, line.split()))
                for line in inputfile.readlines()]
    return data
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def readImagesFromfile(filename, num_frames):
    i=0;
    inputImages =[]
    Img=[]
    for i in range(num_frames):
        framenumber="frame"+str(i)+".jpg"
        img = mpimg.imread(filename+framenumber)
        # im = Image.open(filename+framenumber).convert('LA')
        # im.save('greyscale'+str(i) + '.png')
        # im=cv2.imread('greyscale'+str(i)+'.png', 0)
        gray = rgb2gray(img)
        im = cv2.resize(gray, (50, 50))
        z = np.hstack(im)
        inputImages.append(z)
        #list of 1Darrays(frames)
    return inputImages


def LSTMmodel(number_Frames, batch_x, batch_y):
    learning_rate = 0.001
    n_classes = 80
    num_units = 400
    # Create a saver object which will save all the variables
    saver = tf.train.Saver()
#-----------New code hidden layer--------------------------
    # Network parameter
    n_hidden_1 = 256  # 1st layer number of neurons
    n_hidden_2 = 256  # 2nd layer number of neurons
    #multiLayer
    weights = {
        'h1': tf.Variable(tf.random_normal([num_units, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    }
    bias = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    layer_1 = tf.add(tf.matmul(batch_x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
#-------------------------------------------------------------

#------------old weights and biases---------------------------
    # weights = tf.Variable(tf.random_normal([num_units, n_classes]))
    # bias = tf.Variable(tf.random_normal([n_classes]))
#--------------------------------------------------------------

    x = tf.placeholder("float", [None, None, n_input])
    y = tf.placeholder("float", [None, n_classes])

    #time steps = number_Frames so it can read one image of size(100*100) in one time steps so at the end of time steps it will complete reading frames of the clib
    time_steps = number_Frames

    # [batchsize,n_input]
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)
    input = tf.unstack(x, time_steps, 1)
    '''
    The Network
    '''
    #add muti layer lstn to improve accuracy
    lstm_layer = rnn.MultiRNNCell(rnn.BasicLSTMCell(num_units, forget_bias=1), rnn.BasicLSTMCell(num_units, forget_bias=1))

    # use layer2 output istead of input
    outputs, _ = rnn.static_rnn(lstm_layer, layer_2, dtype="float32")
    # converting last output of dimension [batch_size,num_units] to [batch_size,n_classes] by out_weight multiplication
    # (output*weight)+bias
    prediction = tf.matmul(outputs[-1], weights) + bias
    # loss_function
    # the prediction is then compared to the correct class labels. The numerical result of this comparison is called loss.
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    # optimization
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    # model evaluation
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        #session is initializing the variables we created earlier
        sess.run(init)
        sess.run(opt, feed_dict={x: batch_x, y: batch_y})
        acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
        los = sess.run(loss, feed_dict={x: batch_x, y: batch_y})
    return correct_prediction,acc,los


def clipsFixedsize(allClips):
    #print(len(allClips))
    fixedClipsize = []
    longestLen = len(max(allClips,key=len))
    for clip in allClips:
        if(len(clip)<longestLen):
            diff=longestLen-len(clip)
            iter=0
            while iter < diff :
                clip.append(0)
                iter += 1
            fixedClipsize.append(clip)
        else:
            fixedClipsize.append(clip)
    return fixedClipsize, longestLen

def getTestXandTestY():
    datatest=readingDatafromfile("Testdata.txt")
    for x in datatest:
        num_frames = [f for f in listdir("Clips\\Frames\\" + x[0] +"\\"+x[0]+","+str(x[1])) if
                      isfile(join("Clips\\Frames\\" + x[0]+"\\"+x[0]+","+str(x[1]), f))]
        Cliptest = readImagesFromfile("Clips\\Frames\\" + x[0] +"\\"+x[0]+","+str(x[1])+"\\", len(num_frames)-1)
        Cliptest = list(itertools.chain.from_iterable(Cliptest))
        Clipstest.append(Cliptest)
    xtest,numofframestest=clipsFixedsize(Clipstest)
    batch_xtest = np.array([np.array(xi) for xi in xtest])
    classestest = []
    for item in datatest:
        classestest.append(item[2])
    actionstest = tf.one_hot(classestest, depth=80, on_value=1.0, off_value=0.0, axis=1)
    batch_ytest = tf.Session().run(actionstest)
    return batch_xtest,batch_ytest

n_input = 2500
batchsize = 50
Clips=[]
labels= []

#read clip frames and then convert it to 1d array and put it in clips array
#get num-frames in the specified folder
data=readingDatafromfile("5000output.txt")
for x in data:
    num_frames = [f for f in listdir("Clips\\Frames\\"+x[0]+"\\"+x[0]+","+str(x[1])) if
                 isfile(join("Clips\\Frames\\"+x[0]+"\\"+x[0]+","+str(x[1]), f))]
    Clip=readImagesFromfile("Clips\\Frames\\"+x[0]+"\\"+x[0]+","+str(x[1])+"\\", len(num_frames)-1)
    print("film name",x[0])
    Clip = list(itertools.chain.from_iterable(Clip))
    Clips.append(Clip)
    #labels.append(x[1])

init = tf.global_variables_initializer()
iter1 = 0
iter2 = 0
i = 0
trainDataSize = 400
#de kan 15000 msh 3rfa leh?
# while i < 15:
while iter1 < trainDataSize:
        batch_x = []
        _x = []
        #_label= []
        while iter2 < batchsize:
           # print('image size ', len(ImagesPixelsarray), ' iter2 ' , iter2)
            #print("the Clip",Clips[iter2])
            _x.append(Clips[iter2])
            #_label.append(labels[iter2])
        iter2 += 1
        batchsize += 1
        print("here after batch x")
        #make number of frames of that batch fixed
        _x, number_Frames = clipsFixedsize(_x)
        #convert x and y to numpy array so it will work with lstm
        batch_x= np.array([np.array(xi) for xi in _x])
        classes = []
        for item in data:
            classes.append(item[2])
        actions = tf.one_hot(classes, depth=80, on_value=1.0, off_value=0.0, axis=1)
        batch_y = tf.Session().run(actions)
        print('here after batch y')
        batch_x = batch_x.reshape((batchsize-1, number_Frames, n_input))
        correct_prediction,acc,los = LSTMmodel(number_Frames, batch_x, batch_y)
        if iter1 %10==0:
             print("For iter ", iter1)
             print("Accuracy ", acc)
             print("Loss ", los)

        iter1+=1
saver.save(sess, "filteration_model", write_meta_graph=False)
#         # if i == 10:
#         xtest,ytest=getTestXandTestY()
# correct_prediction,acc,los = LSTMmodel(number_Frames, xtest, ytest)
        #     #test here
        #     #save all model with filteration_model name , write_meta_graph=False---> save only if graph chnaged
        #     #saver.save(weight)
        #     saver.save(sess, "filteration_model", write_meta_graph=False)


    #i += 1