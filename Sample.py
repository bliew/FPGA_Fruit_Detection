import numpy as np
import cv2
import pandas as pd
import random
import glob
import os
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

fruit_images = []
labels = []

for fruit_dir_path in glob.glob("TrainingC/*"):
    #print(fruit_dir_path)
    fruit_label = fruit_dir_path.split("/")[-1] #names of folders in "TrainingC"
    #print(fruit_label)
    for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpg")):
        image = cv2.imread(image_path) #load a color image
        #image = cv2.resize(image, (45, 45)) #resize image
        image = cv2.Canny(image,100,200)
        fruit_images.append(image)
        labels.append(fruit_label)
fruit_images = np.array(fruit_images)
labels = np.array(labels)


#create dictionary of the folders in TrainingC
label_to_id_dict = {v:i for i,v in enumerate(np.unique(labels))}
#print(label_to_id_dict)

id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}
#print(id_to_label_dict)

label_ids = np.array([label_to_id_dict[x] for x in labels])
#print(label_ids)

scaler = StandardScaler() #normalize features

#normalize values after flattening images 
images_scaled = [i.flatten() for i in fruit_images]


##pca = PCA(n_components=3)
#pca_result = pca.fit_transform(images_scaled)


#splits data into training and testing set 
X_train, X_test, y_train, y_test = train_test_split(images_scaled, label_ids, test_size=0.15, random_state = 42)
#print(X_train)

#train Random Forest Classifier with the training set 
forest = RandomForestClassifier(n_estimators=10)
forest = forest.fit(X_train, y_train)

#print(X_test)
#predicts classifier on test set 
test_predictions = forest.predict(X_test)
#print(test_predictions)
#print(y_test)
#obtains score of applying random forest classifier on testing set 
precision = accuracy_score(test_predictions, y_test) * 100
print("Accuracy with RandomForest: {0:.6f}".format(precision))


#reads in TestC 
validation_fruit_images = []
validation_labels = []
original_images = [] 
for fruit_dir_path in glob.glob("TestC/*"):
    fruit_label = fruit_dir_path.split("/")[-1]
    for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpg")):
        orig = cv2.imread(image_path)#load a color image
        orig_image = cv2.cvtColor(orig,cv2.COLOR_BGR2RGB)
        #image = cv2.resize(image, (45, 45)) #resize image
        original_images.append(orig_image)
        image = cv2.Canny(orig_image,100,200)

        validation_fruit_images.append(image)
        validation_labels.append(fruit_label)

validation_fruit_images = np.array(validation_fruit_images)

validation_labels = np.array(validation_labels)
c = list(zip(original_images,validation_fruit_images, validation_labels))
random.shuffle(c)
original_images, validation_fruit_images, validation_labels = zip(*c)


validation_label_ids = np.array([label_to_id_dict[x] for x in validation_labels])
validation_images_scaled = [i.flatten() for i in validation_fruit_images]
#validation_pca_result = pca.transform(validation_images_scaled)

svm_model = SVC().fit(X_train,y_train)
trainscore = svm_model.score(X_train,y_train)
svm_predictions = svm_model.predict(validation_images_scaled)
svm_predictions_list = svm_predictions.tolist()
print(svm_predictions[2])
print(svm_predictions_list)
#print(svm_predictions)
svm_testscore = accuracy_score(validation_label_ids, svm_predictions )*100
#print(svm_testscore)
print("Validation Accuracy with SVM: {0:.6f}".format(svm_testscore))

for i in range(0,len(validation_fruit_images)):
    plt.subplot(121),plt.imshow(original_images[i])
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(validation_fruit_images[i],cmap = 'gray')
    plt.title(svm_predictions_list[i]), plt.xticks([]), plt.yticks([])
    plt.show()