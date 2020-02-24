import cv2
import glob
import numpy as np
from matplotlib import pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox


images = [cv2.imread(file) for file in glob.glob("/Users/Breonna/Desktop/fruit/*.jpg")]
edges = [cv2.Canny(img,100,200) for img in images]


def edgeDetection(E):
    result = []
    for i in range(0,len(E)):
        im = plt.subplot(),plt.imshow(E[i],cmap = 'gray')
        im = plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
        plt.savefig('/Users/Breonna/Desktop/fruitedge/edge' + str(i) + '.jpg')


#redL = (170,50,50)
#redH = (180,255,255)

#yellowL  = (20, 100, 100)
#yellowH = (255, 225, 53)


def colorDetection(I):
    image_hsv = []
    L  = (20,100,110)
    H = (40,180,255)
    for j in range(0,len(I)):
        images[j] = cv2.cvtColor(I[j], cv2.COLOR_BGR2RGB)
        image_hsv.append(cv2.cvtColor(I[j], cv2.COLOR_RGB2HSV))
        mask = cv2.inRange(image_hsv[j], L, H)
        result = cv2.bitwise_and(I[j], I[j], mask=mask)
        plt.subplot(1, 2, 1)
        plt.imshow(mask, cmap="gray")
        plt.subplot(1, 2, 2)
        plt.imshow(result)
        plt.savefig('/Users/Breonna/Desktop/fruitcolor/color' + str(j) + '.jpg')



im = images[8]
bbox, label, conf = cv.detect_common_objects(im)
output_image = draw_bbox(im, bbox, label, conf)
plt.imshow(output_image)
plt.show()   


#edgeDetection(edges)
#colorDetection(images)



