import numpy as np
from sklearn import neighbors, datasets, tree, linear_model

from sklearn.externals import joblib
import timeit

from sklearn.model_selection import cross_val_score

def typesOfLetters(x):
    consonants = "BCÇDFGHJKLMNPQRSTVWXYZbcçdfghjklmnpqrstvwxyz"
    countConsonants = 0
    countVogels = 0
    for letter in x:
        if letter not in consonants:
            countVogels += 1
        else:
            countConsonants += 1
    return (countVogels, countConsonants)

def followedVogels(x):
    consonants = "BCÇDFGHJKLMNPQRSTVWXYZbcçdfghjklmnpqrstvwxyz"
    countFollowedVogels = 0 
    maxFollowedVogels = 0
    countFollowedConsonants = 0 
    maxFollowedConsonants = 0    
    flagBeforeVogel = 0
    flagBeforeConsonant = 0
    for letter in x:
        if letter in consonants:
            countFollowedConsonants += 1
            flagBeforeConsonant = 1            
            
            flagBeforeVogel = 0
            if(maxFollowedVogels < countFollowedVogels):
                maxFollowedVogels = countFollowedVogels
            countFollowedVogels = 0
        else:
            flagBeforeConsonants = 0
            if(maxFollowedConsonants < countFollowedConsonants):
                maxFollowedConsonants = countFollowedConsonants
            countFollowedConsonants = 0  
            
            countFollowedVogels += 1
            flagBeforeVogel = 1            
            
    if(maxFollowedVogels < countFollowedVogels):
        maxFollowedVogels = countFollowedVogels
        
    if(maxFollowedConsonants < countFollowedConsonants):
        maxFollowedConsonants = countFollowedConsonants     
    
    return (maxFollowedVogels, maxFollowedConsonants)
                

def numberOfAccents(x):
    countAccents = 0
    accents = "abcçdefghijklmnopqrstuvwxyz ABCÇDEFGHIJKLMNOPQRSTUVWXYZ"
    for letter in x:
        if letter not in accents:
            countAccents += 1
    return countAccents


def asciiSum(x):
    count = 0;
    for letter in x:
        count += ord(letter)
    return count

def features(X):
    F = np.zeros((len(X),5))
    for x in range(0,len(X)):
        F[x,0] = len(X[x])                      #lengh of the word x
        tupleAux = typesOfLetters(X[x])         #(vogels, consonants)
        tupleFollowed = followedVogels(X[x])    #(vogels, consonants)  
        F[x,1] = tupleFollowed[0]               #number of followed consonants
        F[x,2] = tupleAux[0]                    #number of vogels      
        F[x,3] = tupleFollowed[1]               #number of followed vogels
        F[x,4] = asciiSum(X[x])                 #total ASCII sum for the word x
        
    return F     

def mytraining(f,Y):
    n_neighbors = 2
    weights = 'distance'
    #weights = 'uniform'
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf = clf.fit(f, Y)

    return clf

def mytrainingaux(f,Y,par):
    
    return clf

def myprediction(f, clf):
    Ypred = clf.predict(f)

    return Ypred

