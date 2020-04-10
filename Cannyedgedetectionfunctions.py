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
testing_path = 'TestC/*'


def create_model(path):
    fruit_images=[]
    labels=[]
    original_images =[]
    for fruit_dir_path in glob.glob(path):
        fruit_label = fruit_dir_path.split("/")[-1] #names of folders in "TrainingC"
        for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpg")):
            image = cv2.imread(image_path) #load a color image
            orig_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            original_images.append(orig_image)
            image = cv2.Canny(image,200,200)
            fruit_images.append(image)
            labels.append(fruit_label)
    fruit_images = np.array(fruit_images)
    labels = np.array(labels)
    return fruit_images, labels, original_images

def random_forest_test(Xtrain, Xtest, ytrain, ytest):
    #train Random Forest Classifier with the training set 
    forest = RandomForestClassifier(n_estimators=10)
    forest = forest.fit(Xtrain, ytrain)
    #predicts classifier on test set 
    test_predictions = forest.predict(Xtest)
    #obtains score of applying random forest classifier on testing set 
    precision = accuracy_score(test_predictions, ytest) * 100
    print("Accuracy with RandomForest: {0:.6f}".format(precision))


def svm_test(Xtrain, Xtest, ytrain, ytest, imgs, img_labels):
    svm_model = SVC().fit(Xtrain,ytrain)
    trainscore = svm_model.score(Xtrain,ytrain)
    svm_predictions = svm_model.predict(imgs)
    svm_predictions_list = svm_predictions.tolist()
    #print(svm_predictions)
    svm_testscore = accuracy_score(img_labels, svm_predictions )*100
    #print(svm_testscore)
    print("Validation Accuracy with SVM: {0:.6f}".format(svm_testscore))
    return svm_predictions_list

def plotting_classified_images(test_imgs, orig_imgs, orig_labels, predictions_list):
    for i in range(0,len(test_imgs)):
        plt.subplot(121),plt.imshow(orig_imgs[i])
        plt.title('Original Test Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(test_imgs[i],cmap = 'gray')
        plt.title(str(predictions_list[i])+":"+ str(orig_labels[i])), plt.xticks([]), plt.yticks([])
        #plt.show()
        plt.savefig("Output/"+str(orig_labels[i])+str(i))


if __name__ == "__main__":
    #create training model
    train_result_list = create_model(training_path)
    training_fruit_list = train_result_list[0]
    training_labels_list = train_result_list[1]

    #create testing model
    test_result_list = create_model(testing_path)
    test_fruit_list = test_result_list[0]
    test_labels_list = test_result_list[1]
    test_origimage_list = test_result_list[2]

    #create dictionary for training labels
    label_to_id_dict = {v:i for i,v in enumerate(np.unique(training_labels_list))}
    id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}
    
    #convert labels to id values 0 = Mango, 1 = Orange, 2 = Raspberry
    training_label_ids = np.array([label_to_id_dict[x] for x in training_labels_list]) #making labels into numbers
    
    #normalize features
    scaler = StandardScaler() 
    flattened_images = [i.flatten()for i in training_fruit_list]
    #normalize values after flattening images 
    images_scaled = scaler.fit_transform(flattened_images)
    #linear dimensionality reduction, projecting image to lower dimension
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(images_scaled)
    #splits data into training and testing set 
    X_train, X_test, y_train, y_test = train_test_split(pca_result, training_label_ids, test_size=0.15, random_state = 42)

    #perform random forest classification on training set 
    random_forest_test(X_train, X_test, y_train, y_test)


    #shuffle testing images
    c = list(zip(test_origimage_list,test_fruit_list, test_labels_list))
    random.shuffle(c)
    test_origimage_list, test_fruit_list, test_labels_list = zip(*c)

    #convert testing labels to id values 0 = Mango, 1 = Orange, 2 = Raspberry
    testing_label_ids = np.array([label_to_id_dict[x] for x in test_labels_list])
    

    #normalize test images
    testing_images_scaled = scaler.transform([i.flatten() for i in test_fruit_list])
    testing_pca_result = pca.transform(testing_images_scaled)

    predict_list = svm_test(X_train, X_test, y_train, y_test, testing_pca_result, testing_label_ids)
    
    #get original labels 
    orig_labels = [id_to_label_dict[x] for x in predict_list]
    print(orig_labels)

    plotting_classified_images(test_fruit_list, test_origimage_list, orig_labels, predict_list)

    print("done")



