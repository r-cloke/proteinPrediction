# -*- coding: utf-8 -*-
"""
@author: Ryan Cloke
"""
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
import matplotlib
import random
import scipy.stats
import pickle
import os

path = os.getcwd()

def binaryEncode(peptide):
    """
    A	Alanine
    R	Arginine
    N	Asparagine
    D	Aspartic acid
    C	Cysteine
    Q	Glutamine
    E	Glutamic acid
    G	Glycine
    H	Histidine
    I	Isoleucine
    L	Leucine
    K	Lysine
    M	Methionine
    F	Phenylalanine
    P	Proline
    S	Serine
    T	Threonine
    W	Tryptophan
    Y	Tyrosine
    V	Valine
    """

    #do 1 hot encoding
    binaryPeptide=''
    for aa in peptide:
        binaryAmino=''
        if aa =='A':
            binaryAmino='10000000000000000000'
        if aa =='R':
            binaryAmino='01000000000000000000'
        if aa =='N':
            binaryAmino='00100000000000000000'
        if aa =='D':
            binaryAmino='00010000000000000000'
        if aa =='C':
            binaryAmino='00001000000000000000'
        if aa =='Q':
            binaryAmino='00000100000000000000'
        if aa =='E':
            binaryAmino='00000010000000000000'
        if aa =='G':
            binaryAmino='00000001000000000000'
        if aa =='H':
            binaryAmino='00000000100000000000'
        if aa =='I':
            binaryAmino='00000000010000000000'
        if aa =='L':
            binaryAmino='00000000001000000000'
        if aa =='K':
            binaryAmino='00000000000100000000'
        if aa =='M':
            binaryAmino='00000000000010000000'
        if aa =='F':
            binaryAmino='00000000000001000000'
        if aa =='P':
            binaryAmino='00000000000000100000'
        if aa =='S':
            binaryAmino='00000000000000010000'
        if aa =='T':
            binaryAmino='00000000000000001000'
        if aa =='W':
            binaryAmino='00000000000000000100'
        if aa =='Y':
            binaryAmino='00000000000000000010'
        if aa =='V':
            binaryAmino='00000000000000000001'
        binaryPeptide=binaryPeptide +binaryAmino
        if len(binaryPeptide) == 500*20:
            break            
        
    while len(binaryPeptide) < 500*20:
        binaryPeptide = binaryPeptide +str(0)
            
    binaryPeptide = np.array(list(binaryPeptide),dtype=float)
    binaryPeptide = np.reshape(binaryPeptide,(binaryPeptide.shape[0],1))
    binaryPeptide = np.transpose(binaryPeptide)
    return binaryPeptide
     

def getXData():
    pepLst=[]
    pathwayLst=[]
    seqLst=[]
    
    #randomize input file order
    lines = open(path+'\\protein_seq_pathway.tsv').readlines()[1:]
    random.shuffle(lines)

    #for i in range(len(125957)):
    for i in range(10000):
        seq = lines[i].split('\t')[6].strip('\n')
        seqLst.append(seq)
        
        binaryPeptide = binaryEncode(seq)
        pepLst.append(binaryPeptide)
            
        pathway = lines[i].split('\t')[9].strip('\n')
        pathwayLst.append(pathway)

    X = np.array(pepLst) 
    X = X[:,0, :]
    Y =np.transpose(np.array(pathwayLst))
    Y = np.reshape(Y,(Y.shape[0],1))
    return X,Y,seqLst

(Xdataset, Ydataset, seqLst) = getXData()

le = LabelEncoder()
labels = le.fit_transform(Ydataset)

(trainData, testData, trainLabels, testLabels) = train_test_split(
	Xdataset, labels, test_size=0.20, random_state=42)

model = SGDClassifier(loss="log", random_state=967, n_iter=10)
model.fit(trainData, trainLabels)
 
# evaluate the classifier
print("[INFO] evaluating classifier...")
predictions = model.predict(testData)
print(classification_report(testLabels, predictions,
	target_names=le.classes_))


with open('test_output.csv','w') as fout:
    for i in range(len(testLabels)):
        fout.write(str(seqLst[i])+','+str(le.classes_[testLabels[i]])+'\n')
fout.close()

# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

###############make predictions on unknown proteins
lines = open(path+'\\proteins_with_unknown_function.csv').readlines()[1:]
random.shuffle(lines)

#for i in range(len(125957)):
seqLst=[]
pepLst=[]
for i in range(10600):
    seq = lines[i].split(',')[5].strip('\n')
    seqLst.append(seq)
        
    binaryPeptide = binaryEncode(seq)
    pepLst.append(binaryPeptide)

    Xpred = np.array(pepLst) 
    Xpred = Xpred[:,0, :]

print("making predictions on unknown proteins...")
predictions = model.predict(Xpred)

with open('predictions.csv','w') as fout:
    for i in range(len(predictions)):
        fout.write(str(seqLst[i])+','+str(le.classes_[predictions[i]])+'\n')
fout.close()
