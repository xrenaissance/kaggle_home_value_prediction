#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from sklearn.cross_validation import StratifiedKFold
from collections import defaultdict



################################################################################################
# Copy from KazAnova's starter code
def convert_dataset_to_avg(xc, yc, xt, rounding=2, cols=None):
    xc = xc.tolist()
    xt = xt.tolist()
    yc = yc.tolist()
    if cols == None:
        cols =[k for k in range(0,len(xc[0]))]
    woe = [[0.0 for k in range(0,len(cols))] for g in range(0,len(xt))]
    good = []
    bads = []
    for col in cols:
        dictsgoouds = defaultdict(int)
        dictsbads = defaultdict(int)
        good.append(dictsgoouds)
        bads.append(dictsbads)
    total_count = 0.0
    total_sum = 0.0

    for a in range (0,len(xc)):
        target = yc[a]
        total_sum += target
        total_count += 1.0
        for j in range(0, len(cols)):
            col = cols[j]
            good[j][round(xc[a][col], rounding)] += target
            bads[j][round(xc[a][col], rounding)] += 1.0
    #print(total_goods,total_bads)

    for a in range (0,len(xt)):
        for j in range(0,len(cols)):
            col = cols[j]
            if round(xt[a][col],rounding) in good[j]:
                 woe[a][j] = float(good[j][round(xt[a][col],rounding)]) / float(bads[j][round(xt[a][col],rounding)])
            else :
                 woe[a][j] = round(total_sum/total_count)
    return woe


def convert_to_avg(X,y, Xt, seed=1, cvals=5, roundings=2, columns=None):
    if columns==None:
        columns=[k for k in range(0,(X.shape[1]))]
    #print("it is not!!")
    X=X.tolist()
    Xt=Xt.tolist()
    woetrain=[ [0.0 for k in range(0,len(X[0]))] for g in range(0,len(X))]
    woetest=[ [0.0 for k in range(0,len(X[0]))] for g in range(0,len(Xt))]

    kfolder=StratifiedKFold(y, n_folds=cvals,shuffle=True, random_state=seed)
    for train_index, test_index in kfolder:
        # creaning and validation sets
        X_train, X_cv = np.array(X)[train_index], np.array(X)[test_index]
        y_train = np.array(y)[train_index]

        woecv=convert_dataset_to_avg(X_train,y_train,X_cv, rounding=roundings,cols=columns)
        X_cv=X_cv.tolist()
        no=0
        for real_index in test_index:
            for j in range(0,len(X_cv[0])):
                woetrain[real_index][j]=X_cv[no][j]
            no+=1
        no=0
        for real_index in test_index:
            for j in range(0,len(columns)):
                col=columns[j]
                woetrain[real_index][col]=woecv[no][j]
            no+=1
    woefinal=convert_dataset_to_avg(np.array(X),np.array(y),np.array(Xt), rounding=roundings,cols=columns)

    for real_index in range(0,len(Xt)):
        for j in range(0,len(Xt[0])):
            woetest[real_index][j]=Xt[real_index][j]

    for real_index in range(0,len(Xt)):
        for j in range(0,len(columns)):
            col=columns[j]
            woetest[real_index][col]=woefinal[real_index][j]

    return np.array(woetrain), np.array(woetest)
