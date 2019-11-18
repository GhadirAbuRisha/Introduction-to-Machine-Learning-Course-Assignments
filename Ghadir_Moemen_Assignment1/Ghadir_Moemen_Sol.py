
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 11:23:43 2018

@author: ghadir
"""
# %% Initialization and loading data:

import numpy as np
import matplotlib.pyplot as plt

from scipy import misc
import glob
from sklearn.metrics import confusion_matrix


#---------------------------------------------------------------------------

#Initialization:

n_classes=10    #Number of classes
n_itrs=5000   #Number of itertions
x_train=[]      #Training data set
x_val=[]        #Validation data set
x_test=[]


def get_numericals(string):  #A function used to sort the loaded data to match with labels
    x = string[string.rindex("/")+1:-4]
    return int(x)


#---------------------------------------------------------------------------

#Loading Data:
 
trainig_files=glob.glob ("/home/ghadir/Downloads/ML Course/CIT651 - Assignment 1/Assignment 1 Dataset/Train/*.jpg")
trainig_files.sort(key=get_numericals)

for i in trainig_files:    
    img=misc.imread(i)
    img=img.reshape(784,)
    x_train.append(img)
    
x_train=np.append(x_train, np.ones((len(x_train),1)), axis=1)     #Adding a column of ones for the bias term          
#Loading labels:
training_labels=np.loadtxt("/home/ghadir/Downloads/ML Course/CIT651 - Assignment 1/Assignment 1 Dataset/Train/Training Labels.txt", delimiter="\n")


validation_files=glob.glob ("/home/ghadir/Downloads/ML Course/CIT651 - Assignment 1/Assignment 1 Dataset/Validation/*.jpg")
validation_files.sort(key=get_numericals)

for i in validation_files:    
    img=misc.imread(i)
    img=img.reshape(784,)
    x_val.append(img) 
          
x_val=np.append(x_val, np.ones((len(x_val),1)), axis=1)     #Adding a column of ones for the bias term
#Loading labels:
validation_labels=np.loadtxt("/home/ghadir/Downloads/ML Course/CIT651 - Assignment 1/Assignment 1 Dataset/Validation/Validation Labels.txt", delimiter="\n")




testing_files=glob.glob ("/home/ghadir/Downloads/ML Course/CIT651 - Assignment 1/Assignment 1 Dataset/Test/*.jpg")
testing_files.sort(key=get_numericals)

for i in testing_files:    
    img=misc.imread(i)
    img=img.reshape(784,)
#    img = np.append(img,1)
    x_test.append(img)           
x_test=np.append(x_test, np.ones((len(x_test),1)), axis=1)      #Adding a column of ones for the bias term
#Loading labels:
testing_labels=np.loadtxt("/home/ghadir/Downloads/ML Course/CIT651 - Assignment 1/Assignment 1 Dataset/Test/Test Labels.txt", delimiter="\n")


print ("\n Data loaded! \n")



# %% Perceprton Training:

def perceptron_train():
    print("\n Training started: \n")
    for i in range (10): #Looping over absolute powers of learning rate.
        eta = 10**-i
        w = np.zeros((len(x_train[0]), n_classes)) #Making 10 classifiers for each eta all in one matrix.
        w[0,:]=1    #Setting initial weights
        w_t = np.transpose(w)
        print ("Eta of power -{}".format(i))
    
        error_sum = np.zeros(n_classes)
        for itr in range (n_itrs):      #Number of iterations for the perceprton to stop training
            for m in range(len(x_train)):       #Looping over training data
                image = x_train[m]
                y = np.matmul(w_t,image)        #Getting decision for 10 classes.
                image_label = int(training_labels[m]) #Loading label for current image.
                t = -1*np.ones(n_classes)         #Making array of decisions for all classes based on the given labels.
                t[image_label]=1                  #Setting the value for the right class to one, also from the labels.
    
                for j in range(n_classes):   #Looping over each class to update weights.
                    error = -1*y[j]*t[j]
                    if error > 0:       #If the point is misclassified, update weights
                        error_sum[j] += error
                        w_t[j] = np.add(w_t[j], eta* image *t[j])       #Updating the weights.
            globals()['w%s' % i] = np.transpose(w_t)    #Storing weights of different etas for all 10 classifiers, names: w0, w1, w2 ... to w9.
        
    print ("\n Training done! \n")
    
perceptron_train()
 
# %% Perceptron Testing:
def perceptron_test (data, weights): #Function to test on certain data group with trained 10 classifiers
    k=0
    predictions = np.zeros(len(data))
    for image in data:
        y = np.matmul(np.transpose(weights),image)        #Getting decision for 10 classes
        predictions[k] = np.argmax(y)   #Correct class is the one with max probability, put it in an array of predictions to make confusion matrix
        k+=1    #Increase the counter in prediction to save the prediction of the next image in its corrisponding place
    
    Confusion=confusion_matrix(testing_labels, predictions, labels= np.arange(10))
    return Confusion

def get_CMs(data, weights, name, draw):     #Function to get the confusion matrix plot and draw it or not based on the parameter draw.
        Confusion= perceptron_test(data, weights)
        Accuracy = (np.trace(Confusion)/float(len(data))) *100.0
        if draw:
            fig = plt.figure()
            plt.matshow(Confusion) #
            plt.title("Confusion Matrix of {} \n Acurracy = {}%\n".format(name, Accuracy))
            plt.colorbar()
            plt.ylabel('True Label')
            plt.xlabel('Predicated Label')
            plt.savefig("/home/ghadir/Downloads/ML Course/CIT651 - Assignment 1/Ghadir_Moemen_Assignment1/Confusion-{}.jpg".format(name))
            print ("Accuracy: {} \n".format(Accuracy))
            plt.imshow(Confusion) #
        else:
            print ("Validation")
        
        print ("\n Testing of eta power {} done: \n".format(name))
        print(Confusion)



for i in range (10):      #Drawing confusion matrices for 10 etas on test data for part a. 
    print("\n Testing for part a started: \n")
    get_CMs(x_test, globals()['w%s' % i], i, 1)


    
                ################ End of part a ##############
                
#%% Part b:

print("\n Part b started: \n")

for i in range (10):   #Getting confusion matrices for the validation data set to fine tune the parameter eta.    
    get_CMs(x_val, globals()['w%s' % i], i, 0)

def find_best_etas():   #Function to make an array of the best etas for each digit based on accuracy.
    print("\n Finding best etas ... \n")
    best_etas = np.zeros(10,)
    best_accuracies = np.zeros(10,)
    for i in range(10):
        cm= perceptron_test(x_val, globals()['w%s' % i])
        print(globals()['w%s' % i])
        new_accuracies= np.sum(cm, axis=0)/20

        for j in range(10):
            if new_accuracies[j]>best_accuracies[j]:
                best_accuracies[j]=new_accuracies[j]
                best_etas[j]=i
    return best_etas



#%%
w_b = np.array([])  #Making an empty weight vector to put the best weights in it.
etas = find_best_etas()     #An array of best values of eta for each digit.

print("\n Making combined weight vector ... \n")
for i in range (10):    #Looping to add the best weights to the new weight vector.
    m = int(etas[i])
    print(m)
    w_b=np.append(w_b, globals()['w%s' % m][:,i])

w_b=w_b.reshape(10, 785)
w_b=np.transpose(w_b)

print("\n Getting confusion matrix for combined weight vector ... \n")
get_CMs(x_test, w_b, "b", 1)    #Testing the performance of the combined weight vector on the test data.

print("\n Program ended! \n\n")


                ################ End of part b ##############