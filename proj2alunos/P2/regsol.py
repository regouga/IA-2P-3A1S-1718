import numpy as np
from sklearn import datasets, tree, linear_model
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_score
import timeit

def mytraining(X,Y):
    """
    # Kernel Ridge Regression
    reg = KernelRidge(kernel='rbf', gamma=0.1,alpha=0.0003)
    reg.fit(X,Y)
    Ykr = reg.predict(X)
    """
    reg = KernelRidge(kernel='polynomial', gamma=0.0003, alpha=0.1)
    reg.fit(X,Y)
    Ykr = reg.predict(X)   
    
    return reg
    
def mytrainingaux(X,Y,par):
    
    reg.fit(X,Y)
                
    return reg

def myprediction(X,reg):

    Ypred = reg.predict(X)

    return Ypred


def secondRegressionMethod(X,Y):
    reg = KernelRidge(kernel='polynomial', gamma=1, alpha=0.1)
    reg.fit(X,Y)
    Ykr = reg.predict(X)
    
    return reg