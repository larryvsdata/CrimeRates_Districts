# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 18:30:02 2018

@author: Erman
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import math
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error



class CrimeDistrictAnalysis():
    
    def __init__(self):
        self.df=pd.read_csv("formattedData.csv")
        self.X=self.df.drop(['CRate','VCRate'], axis=1).values.tolist()
        self.y=self.df[['VCRate']].values.tolist()
        self.max_iter=1000000
        self.clf=RandomForestRegressor(n_estimators=100)
        self.model=None
        self.splitRatio=0.33
        self.trainX=[]
        self.trainY=[]
        self.testX=[]
        self.testY=[]
        self.validationErrors=[]
        self.kFold=5
        
        self.models=[]
        self.modelType=2
        self.degree=3
        self.label=1
        
    def getInfo(self):
        self.label=int(input("Enter the number of the label to be predicted. 1 or 2 : "))
#        self.modelType=int(input("Enter 1 for a Linear Model and 2 for a Non-Linear model: "))
#        
#        if self.modelType ==2:
#            self.degree=int(input("Enter the degree of the NL model (2-3) : "))
            
#            if self.degree ==2:
#                self.max_iter=100000
#            else:
#                self.max_iter=7500
 
        
    def trainTestSplit(self):
        if self.label==1:
            self.y=self.df[['CRate']].values.tolist()
            
        elif self.label==2:
            self.y=self.df[['VCRate']].values.tolist()
        self.trainX, self.testX,self.trainY, self.testY = train_test_split(self.X, self.y, test_size=self.splitRatio, random_state=423)
    
    def trainAndValidate(self):
        
            self.model.fit(self.trainX,self.trainY)
            validationRatio=1/self.kFold
            
            for validation in range(self.kFold):
               clf=RandomForestRegressor(n_estimators=100)
               self.trainX, self.validateX,self.trainY, self.validateY = train_test_split(self.trainX, self.trainY, test_size=validationRatio)
               clf.fit(self.trainX,self.trainY)
               outcome=clf.predict(self.validateX)
                   
               self.validationErrors.append(mean_squared_error(outcome,self.validateY))
               self.models.append(clf)
        
    # Choose the model that is the least biased of all validated models.        
            self.model=self.models[self.validationErrors.index(min(self.validationErrors))]
    
    # Release the memory
            del self.models[:]

    def test(self):
        self.results=self.model.predict( self.testX)
        self.finalError=mean_squared_error(self.results,self.testY)
        
        
    def fixTheModel(self):
#        if self.modelType==1:
            self.model=self.clf
#        elif self.modelType==2:
#            self.model=make_pipeline(PolynomialFeatures(self.degree), self.clf)
        
    def plotTheResult(self):
        
        modelName='Crime rate Modeling'

        
        plt.plot(self.results,'r--', label="Model Results ")
        plt.plot(self.testY, 'bs', label="Real Values ")
        plt.legend(loc='best')
        plt.title(modelName)
        plt.show()
        
    def printResults(self):
       
       for ii in range(len(self.results)):
           print(self.testY[ii],self.results[ii])    
    def report(self):
        
        modelName=""
        ending=""
        if self.modelType==1:
            modelName="Linear"
        else:
            modelName="Non-linear"
            ending=" of "+ str(self.degree)+ " degree."
        
         
        print(modelName+" model "+ ending)    
        
        print(str(self.kFold)+" fold validation errors: ")
        print(self.validationErrors)
            
        print("Overall error is: ",self.finalError )
        
        
        
    def plot_coefficients(self):
        coef = self.model.feature_importances_
 
         # create plot
        importances = pd.DataFrame({'feature':self.df.drop(['CRate','VCRate'], axis=1).columns.values,'importance':np.round(coef,3)})
        importances = importances.sort_values('importance',ascending=True).set_index('feature')
        print( importances)
        importances.plot.barh() 
            
        
if __name__ == '__main__':
    
    analysis=CrimeDistrictAnalysis()
    analysis.getInfo()
    analysis.trainTestSplit()
    analysis.fixTheModel()
    analysis.trainAndValidate()
    analysis.test()
    analysis.plotTheResult()
    analysis.printResults()
    analysis.report()
    analysis.plot_coefficients()
    
    
    