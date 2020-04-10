import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import glob
import os
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

#performs edgedetection on each folder of images which it is passed as a parameter
def edgeDetection(path):
    fruitlist = os.listdir(path)
    for f in fruitlist[0:1] :
        fruitpath = '%s/%s' % (path,f)
        imagelist = os.listdir(fruitpath)
        for i in imagelist:
            imagepath = '%s/%s' % (fruitpath,i)
            image = cv2.imread(imagepath)
            edges = cv2.Canny(image,100,200)
            plt.subplot(121),plt.imshow(image,cmap = 'gray')
            plt.title('Original Image'), plt.xticks([]), plt.yticks([])
            plt.subplot(122),plt.imshow(edges,cmap = 'gray')
            plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
            #saves detected image in folder called "Edge Detection" 
            #saved with its corresponding fruit
            plt.savefig('Edge_detection/'+ f+"/"+ str(i))


fruit_images = []
labels = [] 
for fruit_dir_path in glob.glob("TrainingC/*"):
    #print(fruit_dir_path)
    fruit_label = fruit_dir_path.split("/")[-1] #names of folders in "TrainingC"
    #print(fruit_label)
    for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpg")):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR) #load a color image
        image = cv2.resize(image, (45, 45)) #resize image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #color conversion to BGR
        
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
images_scaled = scaler.fit_transform([i.flatten() for i in fruit_images])

pca = PCA(n_components=3)
pca_result = pca.fit_transform(images_scaled)

print(pca_result)

# # tsne = TSNE(n_components=2, perplexity=40.0)
# # tsne_result = tsne.fit_transform(pca_result)
# # tsne_result_scaled = StandardScaler().fit_transform(tsne_result)

#splits data into training and testing set 
X_train, X_test, y_train, y_test = train_test_split(pca_result, label_ids, test_size=0.25, random_state=40)

# #print(X_test)

#train Random Forest Classifier with the training set 
forest = RandomForestClassifier(n_estimators=10)
forest = forest.fit(X_train, y_train)

#predicts classifier on test set 
test_predictions = forest.predict(X_test)
#obtains score of applying random forest classifier on testing set 
precision = accuracy_score(test_predictions, y_test) * 100
print("Accuracy with RandomForest: {0:.6f}".format(precision))


#reads in TestC 
validation_fruit_images = []
validation_labels = [] 
for fruit_dir_path in glob.glob("TestC/*"):
    fruit_label = fruit_dir_path.split("/")[-1]
    for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpg")):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        image = cv2.resize(image, (45, 45))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        validation_fruit_images.append(image)
        validation_labels.append(fruit_label)


validation_fruit_images = np.array(validation_fruit_images)
validation_labels = np.array(validation_labels)
validation_label_ids = np.array([label_to_id_dict[x] for x in validation_labels])
validation_images_scaled = scaler.transform([i.flatten() for i in validation_fruit_images])
validation_pca_result = pca.transform(validation_images_scaled)


#test with Random Forest
test_predictions = forest.predict(validation_pca_result)
precision = accuracy_score(validation_label_ids, test_predictions) * 100
print("Validation Accuracy with Random Forest: {0:.6f}".format(precision))


#test with SVM Model
svm_model = SVC().fit(X_train,y_train)
trainscore = svm_model.score(X_train,y_train)
svm_predictions = svm_model.predict(validation_pca_result)
svm_testscore = accuracy_score(validation_label_ids, svm_predictions )*100
print("Validation Accuracy with SVM: {0:.6f}".format(svm_testscore))
#print('SVM','Training score: {:.3f}\nTest score: {:.3f}'.format(trainscore,testscore)) 


#test with K-Nearest Neighbors
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)
k_predictions = classifier.predict(validation_pca_result)
k_testscore = accuracy_score(validation_label_ids, k_predictions )*100
print("Validation Accuracy with K-Nearest Neighbors: {0:.6f}".format(k_testscore))
# print(confusion_matrix(validation_label_ids, y_pred))
# print(classification_report(validation_label_ids, y_pred))

#edgeDetection('TrainingC')
