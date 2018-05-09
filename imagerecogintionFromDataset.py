import tensorflow as tf
from tensorflow.contrib import rnn
import keras
from keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import cv2
def readImagesFromfile():
    i=0;
    inputImages =[]
    Img=[]
    for i in range(80):
        framenumber="frame"+str(i)+".jpg"
        im = Image.open("C:\\Users\\hassan ali\\Downloads\\python-projects\\Imagerecognition\\clip1\\"+framenumber).convert('LA')
        im.save('greyscale' +str(i) + '.png')
        im=cv2.imread('greyscale'+str(i)+'.png', 0)
        im = cv2.resize(im, (100, 100))
        z = np.hstack(im)
        inputImages.append(z)
    return inputImages

ImagesPixelsarray=readImagesFromfile()
time_steps=100
n_input=100
batchsize=10
learning_rate=0.001
n_classes=1
num_units=128
weights=tf.Variable(tf.random_normal([num_units,n_classes]))
bias=tf.Variable(tf.random_normal([n_classes]))
x=tf.placeholder("float",[None,time_steps,n_input])
y=tf.placeholder("float",[None,1])
#[batchsize,n_input]
# Required shape: 'timesteps' tensors list of shape (batch_size, n_input)
input=tf.unstack(x ,time_steps,1)
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

#model evaluation
correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#initialize variables
b_size = 10
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    iter1=0
    iter2 = 0
    while iter1<8:
        batch_x=[]
        _x=[]
        while iter2<batchsize:
           # print('image size ', len(ImagesPixelsarray), ' iter2 ' , iter2)
            _x.append(ImagesPixelsarray[iter2])
            iter2+=1
        batchsize+=10
        batch_x= np.array([np.array(xi) for xi in _x])
        #print('batch numpy array',batch_x)
        batch_x = batch_x.reshape((10, time_steps, n_input))
        n = 10
        _y = np.c_[np.random.randint(0, 2, (n))]
        batch_y=np.array([np.array(yi) for yi in _y])
        sess.run(opt, feed_dict={x: batch_x, y: batch_y})
        if iter1 %2==0:
             acc=sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
             los=sess.run(loss,feed_dict={x:batch_x,y:batch_y})
             print("For iter ",iter1)
             print("Accuracy ",acc)
             print("Loss ",los)
             print("__________________")
        iter1+=1
