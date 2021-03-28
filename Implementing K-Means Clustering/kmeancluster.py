import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rd
import csv

train_set = []
with open('data_k_mean.txt','r') as file:
    new_reader = csv.reader(file,delimiter=' ')
    for row in new_reader:
        train_set.append(row)

for i in range(len(train_set)):
    for j in range(len(train_set[i])):
        train_set[i][j] = float(train_set[i][j])

x = []
for train in train_set:
    x.append(train[0])
y = []
for train in train_set:
    y.append(train[1])

train_set_array = np.array(train_set)

plt.plot(train_set_array[:,0:1],train_set_array[:, 1:], linestyle = '', marker='o', color='k')
plt.show()

X = pd.DataFrame({'x': x,
                 'y': y})

k = int(input("Enter the Value of K: "))
Centroids = (X.sample(n=k))
Centroids

diff = 1
j=0

while(diff!=0):
    XD=X
    i=1
    for index1,row_c in Centroids.iterrows():
        ED=[]
        for index2,row_d in XD.iterrows():
            d1=(row_c["x"]-row_d["x"])**2
            d2=(row_c["y"]-row_d["y"])**2
            d=np.sqrt(d1+d2)
            ED.append(d)
        X[i]=ED
        i=i+1

    C=[]
    for index,row in X.iterrows():
        min_dist=row[1]
        pos=1
        for i in range(k):
            if row[i+1] < min_dist:
                min_dist = row[i+1]
                pos=i+1
        C.append(pos)
    X["Cluster"]=C
    Centroids_new = X.groupby(["Cluster"]).mean()[["x","y"]]
    if j == 0:
        diff=1
        j=j+1
    else:
        diff = (Centroids_new['x'] - Centroids['x']).sum() + (Centroids_new['y'] - Centroids['y']).sum()
    Centroids = X.groupby(["Cluster"]).mean()[["x","y"]]

X.head()
X.tail()

color=['green','red','cyan']
for l in range(k):
    data=X[X["Cluster"]==l+1]
    plt.scatter(data["x"],data["y"],c=color[l])
plt.show()