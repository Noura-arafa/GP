import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from Model.LSTMmodel import lstmModel
#"E:\FOURTH YEAR\GP\youTube control\kiss session\epoch 14",
class oneAgainstother:
    LSTMmodel = lstmModel()
    types = str,int,int

    def readingDatasetfromfile(self, filename):
        with open(filename, "r") as inputfile:
            data = [tuple(t(e) for t, e in zip(self.types, line.split()))
                    for line in inputfile.readlines()]
        return data

    def oneClassagainstother(self):
        Accuracy = []
        Predictions = []
        filepath="E:\FOURTH YEAR\GP\youTube control\\test one against other.txt"
        arrayfilepath="E:\FOURTH YEAR\GP\DataSet\one agiants\oneAgainstnumbyarrays\\"
        data = self.readingDatasetfromfile(filepath)
        modelpath=[
            "E:\FOURTH YEAR\GP\youTube control\kiss session\epoch 14",
            "E:\FOURTH YEAR\GP\youTube control\dance session\epoch 14",
            "E:\FOURTH YEAR\GP\youTube control\eat session\epoch 14",
            "E:\FOURTH YEAR\GP\youTube control\hug session\epoch 14",
            "E:\FOURTH YEAR\GP\youTube control\smoke session\epoch 14"
                   ]
        allprediction = self.LSTMmodel.testOneAgainst(filepath, arrayfilepath,modelpath )
        Predictionsindex=[]
        sizedata=len(allprediction[0])
        count = 0
        for i in range(sizedata):
            predictionforoneclip =[]
            for j in range(4):
                pred = allprediction[j][i]
                predictionforoneclip.append(pred)
            maxvalue = max(predictionforoneclip)
            index = predictionforoneclip.index(maxvalue)
            print("Class ",index+1)
            if (data[i][2] == index+1):
                count+=1
        print(count/sizedata)
x = oneAgainstother()
x.oneClassagainstother()