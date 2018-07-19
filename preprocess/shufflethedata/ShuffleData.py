# import random
#
import tensorflow as tf
import numpy as np
types = str, int ,int
with open("E:\\FourthYear\\3-GP\\GP-Git\\DataProcess\\shufflethedata\\10000output.txt", "r") as inputfile:
    data = [tuple(t(e) for t,e in zip(types, line.split()))
                for line in inputfile]
# print(len(data))
#
# random.shuffle(data)
#
# thefile = open('10000output.txt', 'w')
# count=0;
# limit=10000
# for item in data:
#   if(count==limit and limit!=3000):
#       thefile = open('Testdata.txt', 'w')
#       count=0
#       limit=3000
#   if(count ==limit and limit==3000):
#     break;
#   thefile.write("%s   " % item[0])
#   thefile.write('%d   ' % item[1])
#   thefile.write('%d' % item[2])
#   thefile.write("\n")
#   count+=1


actions = tf.one_hot(List, depth=80, on_value=1.0, off_value=0.0, axis=1)
actions=tf.Session().run(actions)
print(actions)

