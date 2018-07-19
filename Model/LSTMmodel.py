import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

from Dataprocess.dataPreprocess import dataPreprocess


class lstmModel:
    #train and test data attributes
    Clips = []
    labels = []
    maxnumberframes = 90
    epoch = 0
    epochs = 1
    trainDataSize = 1
    batchsize = 50
    batchlimit = 50
    dataIterator = 0
    batchIterator = 0
    model_path = "./sessions/"
    datapreprocess = dataPreprocess()

    #model attributes
    learning_rate = 0.001
    n_classes = 2
    num_units = 256
    n_input = 10000
    weights = tf.Variable(tf.random_normal([num_units, n_classes]))
    bias = tf.Variable(tf.random_normal([n_classes]))
    x = tf.placeholder("float", [None, None, n_input])
    y = tf.placeholder("float", [None, n_classes])
    numlayer = 5

    def getTrainData(self,dataFile,arrayFilepath):
        data = self.datapreprocess.readingDatasetfromfile(dataFile)
        for item in data:
            Clip = dataPreprocess.readingClipsarrayfromfile(arrayFilepath + item[0] + ',' + str(item[1]) + '.txt')
            self.Clips.append(Clip)
            self.labels.append(item[2])
        self.Clips = dataPreprocess.paddingbatchclips(self.Clips, self.maxnumberframes * 100 * 100)
        return self.Clips,self.labels

    def make_cell(self,lstm_size):
        return tf.nn.rnn_cell.BasicLSTMCell(lstm_size, state_is_tuple=True)

    def trainModel(self,dataFile,arrayFilepath):
        input = tf.unstack(self.x, self.maxnumberframes, 1)
        cell = tf.nn.rnn_cell.MultiRNNCell([self.make_cell(self.num_units) for _ in range(self.numlayer)], state_is_tuple=True)
        outputs, _ = rnn.static_rnn(cell, input, dtype="float32")
        pred = tf.matmul(outputs[-1], self.weights) + self.bias

        # loss_function
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=self.y))
        # optimization
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

        # model evaluation
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init)
            trainClips,trainLabels = self.getTrainData(dataFile,arrayFilepath)
            while self.epoch < self.epochs:
                while self.dataIterator < self.trainDataSize:
                    batch_x = trainClips[self.batchIterator:self.batchlimit]
                    actions = tf.one_hot(trainLabels[self.batchIterator:self.batchlimit], depth=1, on_value=1.0, off_value=0.0, axis=1)
                    batch_y = tf.Session().run(actions)
                    batch_x = np.array([np.array(xi) for xi in batch_x])
                    batch_x = batch_x.reshape((self.batchsize, self.maxnumberframes, self.n_input))
                    batch_y = np.array([np.array(yi) for yi in batch_y])
                    self. batchIterator = self.batchlimit
                    self.batchlimit += self.batchsize
                    print("dataIterator ", self.dataIterator)
                    acc = sess.run(accuracy, feed_dict={self.x: batch_x, self.y: batch_y})
                    los = sess.run(loss, feed_dict={self.x: batch_x, self.y: batch_y})
                    self.dataIterator += self.batchsize
                    print("Accurecy : ", acc)
                save_path = saver.save(sess, self.model_path + "epoch " + str(self.epoch))
                self.epoch += 1
                self.batchlimit = 50
                self.dataIterator = 0
                self.batchIterator = 0
        return save_path

    def getTestData(self,dataFile,arrayFilepath):
        testData = self.datapreprocess.readingDatasetfromfile(dataFile)
        for item in testData:
            Cliptest = dataPreprocess.readingClipsarrayfromfile(arrayFilepath + item[0] + ',' + str(item[1]) + '.txt')
            self.Clips.append(Cliptest)
            self.labels.append(item[2])
        self.Clips = dataPreprocess.paddingbatchclips(self.Clips, self.maxnumberframes * 100 * 100)
        testactions = tf.one_hot(self.labels[0:len(self.labels)], depth=2, on_value=1.0, off_value=0.0, axis=1)
        test_y = tf.Session().run(testactions)
        test_xReshape = np.array([np.array(xi) for xi in self.Clips])
        test_xReshape = test_xReshape.reshape((len(self.Clips), self.maxnumberframes, self.n_input))
        test_y = np.array([np.array(xi) for xi in test_y])
        return test_xReshape, test_y

    def testModel(self,dataFile,arrayfilepath,modelpath):
        input = tf.unstack(self.x, self.maxnumberframes, 1)
        cell = tf.nn.rnn_cell.MultiRNNCell([self.make_cell(self.num_units) for _ in range(self.numlayer)], state_is_tuple=True)
        outputs, _ = rnn.static_rnn(cell, input, dtype="float32")
        pred = tf.matmul(outputs[-1], self.weights) + self.bias

        # loss_function
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=self.y))
        # optimization
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

        # model evaluation
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init)

            # Restore model weights from previously saved model
            saver.restore(sess, modelpath)
            testx, testy, = self.getTestData(dataFile,arrayfilepath)

            acc= sess.run(accuracy, feed_dict={self.x: testx, self.y: testy})
            prediction = sess.run(pred, feed_dict={self.x: testx, self.y: testy})
        return acc , prediction

    def testOneAgainst(self,dataFile,arrayfilepath,modelpath):
        input = tf.unstack(self.x, self.maxnumberframes, 1)
        cell = tf.nn.rnn_cell.MultiRNNCell([self.make_cell(self.num_units) for _ in range(self.numlayer)], state_is_tuple=True)
        outputs, _ = rnn.static_rnn(cell, input, dtype="float32")
        pred = tf.matmul(outputs[-1], self.weights) + self.bias

        # loss_function
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=self.y))
        # optimization
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

        # model evaluation
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        accuracy =[]
        predictions=[]
        with tf.Session() as sess:
            sess.run(init)
            testx, testy, = self.getTestData(dataFile, arrayfilepath)
            for item in modelpath:
            # Restore model weights from previously saved model
                saver.restore(sess, item)
                prediction = sess.run(pred, feed_dict={self.x: testx, self.y: testy})
                predictions.append(prediction)
        sess.close()
        return  predictions

    def filter(self,Clips,modelpath):
        Clips = dataPreprocess.paddingbatchclips(Clips, self.maxnumberframes * 100 * 100)
        Clipsreshape = np.array([np.array(xi) for xi in Clips])
        Clipsreshape = Clipsreshape.reshape((len(Clips), self.maxnumberframes, self.n_input))
        input = tf.unstack(self.x, self.maxnumberframes, 1)
        cell = tf.nn.rnn_cell.MultiRNNCell([self.make_cell(self.num_units) for _ in range(self.numlayer)],
                                           state_is_tuple=True)
        outputs, _ = rnn.static_rnn(cell, input, dtype="float32")
        pred = tf.matmul(outputs[-1], self.weights) + self.bias
        labels = []
        for i in range(len(Clips)):
            self.labels.append(100)
        testactions = tf.one_hot(self.labels[0:len(self.labels)], depth=2, on_value=1.0, off_value=0.0, axis=1)
        test_y = tf.Session().run(testactions)
        test_y = np.array([np.array(xi) for xi in test_y])
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init)
            # Restore model weights from previously saved model
            saver.restore(sess, modelpath)
            prediction = sess.run(pred, feed_dict={self.x: Clipsreshape, self.y:test_y })
        return  prediction