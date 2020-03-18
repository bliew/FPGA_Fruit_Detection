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
