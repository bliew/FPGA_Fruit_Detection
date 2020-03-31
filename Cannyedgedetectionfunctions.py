import numpy as np
import cv2
import pandas as pd
import random
import glob
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

training_path = 'TrainingC/*'
testing_path = 'TestingC/*'

fruit_images = []
labels = [] 

X_train = None
X_test = None
y_train = None 
y_test = None

def training_model():
    global fruit_images, labels
    for fruit_dir_path in glob.glob("TrainingC/*"):
        fruit_label = fruit_dir_path.split("/")[-1] #names of folders in "TrainingC"
        for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpg")):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR) #load a color image
            #image = cv2.resize(image, (45, 45)) #resize image
            image = cv2.Canny(image,100,200)
            fruit_images.append(image)
            labels.append(fruit_label)
    fruit_images = np.array(fruit_images)
    labels = np.array(labels)

def create_dictionary():
    #create dictionary of the folders in TrainingC
    label_to_id_dict = {v:i for i,v in enumerate(np.unique(labels))}
    id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}
    label_ids = np.array([label_to_id_dict[x] for x in labels]) #making labels into numbers
    return label_ids

def perform_training(labelID):
    scaler = StandardScaler() #normalize features
    flattened_images = [i.flatten()for i in fruit_images]
    #normalize values after flattening images 
    images_scaled = scaler.fit_transform(flattened_images)
    #linear dimensionality reduction, projecting image to lower dimension
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(images_scaled)
    #splits data into training and testing set 
    X_train, X_test, y_train, y_test = train_test_split(pca_result, labelID, test_size=0.15, random_state = 42)
    

def Random_Forest_Test():
    #train Random Forest Classifier with the training set 
    forest = RandomForestClassifier(n_estimators=10)
    forest = forest.fit(X_train, y_train)
    #predicts classifier on test set 
    test_predictions = forest.predict(X_test)
    #obtains score of applying random forest classifier on testing set 
    precision = accuracy_score(test_predictions, y_test) * 100
    print("Accuracy with RandomForest: {0:.6f}".format(precision))




if __name__ == "__main__":
    training_model()
    # for i in fruit_images:
    #     id = 0
    #     plt.subplot(121),plt.imshow(i,cmap = 'gray')
    #     plt.title(str(labels[id])), plt.xticks([]), plt.yticks([])
    #     plt.show()
    #     id = id+1
    ids = create_dictionary()
    training_result = perform_training(ids)
    Random_Forest_Test()


