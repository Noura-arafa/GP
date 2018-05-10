import tensorflow as tf
from tensorflow.contrib import rnn
import keras
from keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import cv2
import itertools
def readImagesFromfile(filename,num_frames,clipnumber):
    i=0;
    inputImages =[]
    Img=[]
    for i in range(num_frames):
        framenumber="frame"+str(i)+".jpg"
        im = Image.open(filename+framenumber).convert('LA')
        im.save('greyscale'+str(clipnumber) +str(i) + '.png')
        im=cv2.imread('greyscale'+str(clipnumber)+str(i)+'.png', 0)
        im = cv2.resize(im, (100, 100))
        z = np.hstack(im)
        inputImages.append(z)
        #list of 1Darrays(frames)
    return inputImages

#1D array of all the frames



Clip1=readImagesFromfile("C:\\Users\\hassan ali\\Downloads\\python-projects\\Imagerecognition\\clip1\\",80,1)
Clip1 = list(itertools.chain.from_iterable(Clip1))
Clip2=readImagesFromfile("C:\\Users\\hassan ali\\Downloads\\python-projects\\Imagerecognition\\clip2\\",88,2)
Clip2 = list(itertools.chain.from_iterable(Clip2))
Clip3=readImagesFromfile("C:\\Users\\hassan ali\\Downloads\\python-projects\\Imagerecognition\\clip3\\",68,3)
Clip3 = list(itertools.chain.from_iterable(Clip3))
Clip4=readImagesFromfile("C:\\Users\\hassan ali\\Downloads\\python-projects\\Imagerecognition\\clip4\\",68,4)
Clip4 = list(itertools.chain.from_iterable(Clip4))
Clips=[]
Clips.append(Clip1)
Clips.append(Clip2)
Clips.append(Clip3)
Clips.append(Clip4)
number_Frames=100
n_input=10000
batchsize=1
learning_rate=0.001
n_classes=1
num_units=128
weights=tf.Variable(tf.random_normal([num_units,n_classes]))
bias=tf.Variable(tf.random_normal([n_classes]))
x=tf.placeholder("float",[None,None,n_input])
y=tf.placeholder("float",[None,1])
#[batchsize,n_input]
# Required shape: 'timesteps' tensors list of shape (batch_size, n_input)
input=tf.unstack(x ,80,1)
print('numberOfframes',number_Frames)
'''
The Network
'''
lstm_layer=rnn.BasicLSTMCell(num_units,forget_bias=1)
outputs,_=rnn.static_rnn(lstm_layer,input,dtype="float32")
#converting last output of dimension [batch_size,num_units] to [batch_size,n_classes] by out_weight multiplication
#(output*weight)+bias
prediction=tf.matmul(outputs[-1],weights)+bias
#loss_function
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
#optimization
opt=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
argmaxprediction=tf.argmax(prediction,1)
#model evaluation
correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
print('cor prediction',correct_prediction)
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#initialize variables
b_size = 10
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
iter1 = 0
iter2 = 0
while iter1<1:
    batch_x=[]
    _x=[]
    while iter2<batchsize:
       # print('image size ', len(ImagesPixelsarray), ' iter2 ' , iter2)
       # print("the Clip",Clips[iter2])
        _x.append(Clips[iter2])
        iter2+=1
    batchsize+=1
    print("X",_x)
    batch_x= np.array([np.array(xi) for xi in _x])
    #print('batch numpy array',batch_x)
    if(iter1==0):
        number_Frames=80
    elif (iter1==1):
        number_Frames=88
    elif (iter1==2):
        number_Frames=68
    else:
        number_Frames=68

    batch_x = batch_x.reshape((1, number_Frames, n_input))
    #batch size
    n = 1
    _y = np.c_[np.random.randint(0, 2, (n))]
    #print('the y',_y)
    batch_y=np.array([np.array(yi) for yi in _y])
    #session.run take the variable to excute its operation
    #ex: z=tf.matmul(x,y) sess.run(z) so it'll calculate
    #the z value
    #here we excute the opt value and send in the
    #feed_dect the placeholders
    sess.run(opt, feed_dict={x: batch_x, y: batch_y})
    #print('outputs ',sess.run(outputs,feed_dict={x:batch_x,y:batch_y}))
    #print('prediction ',sess.run(prediction,feed_dict={x:batch_x,y:batch_y}))
    #print('correct prediction ',sess.run(correct_prediction,feed_dict={x:batch_x,y:batch_y}))
    if iter1 %2==0:
         acc=sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
         los=sess.run(loss,feed_dict={x:batch_x,y:batch_y})
         print("For iter ",iter1)
         print("Accuracy ",acc)
         print("Loss ",los)
         print("__________________")
    iter1+=1
