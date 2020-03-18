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

fruit_images = []
labels = [] 
for fruit_dir_path in glob.glob("TrainingC/*"):
    #print(fruit_dir_path)
    fruit_label = fruit_dir_path.split("/")[-1] #names of folders in "TrainingC"
    #print(fruit_label)
    for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpg")):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR) #load a color image
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
images_scaled = scaler.fit_transform([i.flatten() for i in fruit_images])


pca = PCA(n_components=3)
pca_result = pca.fit_transform(images_scaled)

#splits data into training and testing set 
X_train, X_test, y_train, y_test = train_test_split(pca_result, label_ids, test_size=0.25, random_state=40)


#train Random Forest Classifier with the training set 
forest = RandomForestClassifier(n_estimators=10)
forest = forest.fit(X_train, y_train)


#predicts classifier on test set 
test_predictions = forest.predict(X_test)
#obtains score of applying random forest classifier on testing set 
precision = accuracy_score(test_predictions, y_test) * 100
print("Accuracy with RandomForest: {0:.6f}".format(precision))

plt.scatter(X_train, y_train, color = "red")
#plt.plot(X_train, test_predictions, color = "green")
plt.title("Salary vs Experience (Training set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

