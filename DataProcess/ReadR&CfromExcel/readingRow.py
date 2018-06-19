import pandas as pd
types = str, int ,int
with open("E:\\FourthYear\\3-GP\\GP-Git\\DataProcess\\shufflethedata\\10000clipshuffled.txt", "r") as inputfile:
    data = [tuple(t(e) for t,e in zip(types, line.split()))
                for line in inputfile]

df1 = pd.read_csv("E:\\FourthYear\\3-GP\\GP-Git\\DataProcess\\ReadR&CfromExcel\\Datasetoutput.csv")
thefile = open('output.txt', 'w')
for item in data:
    array = df1.loc[(df1['timestamp'] == item[1]) & (df1['Name'] == item[0])]
    List=array['class'].values.tolist()
    for value in List:
        thefile.write("%s   " % item[0])
        thefile.write('%d   ' % item[1])
        thefile.write('%d' % value)
        thefile.write("\n")

