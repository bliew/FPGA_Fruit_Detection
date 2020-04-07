import cv2
import sys
import numpy
import matplotlib.pyplot as plt

scale = 1
delta = 0
ddepth = cv2.CV_16S

def convert2gray(img_path):
    image = cv2.imread(img_path) #load a color image
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    return gray_image

def gaussianBlur(img):
    blur_image = cv2.GaussianBlur(img,(3,3),0)
    return blur_image

def sobelx(img):
    x_deriv = cv2.Sobel(img, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    abs_x_deriv = cv2.convertScaleAbs(x_deriv)
    return abs_x_deriv

def sobely(img):
    y_deriv = cv2.Sobel(img, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    abs_y_deriv = cv2.convertScaleAbs(y_deriv)
    return abs_y_deriv

def weightedsobel(abs_x, abs_y):
    grad = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
    return grad


if __name__ == "__main__":   
    gray_im = convert2gray("TrainingC/Mango/13_100.jpg")
    gaussblur = gaussianBlur(gray_im)
    grad_x = sobelx(gaussblur)
    grad_y = sobely(gaussblur)
    sobelweight = weightedsobel(grad_x,grad_y)

    plt.imshow(sobelweight, cmap = 'gray')
    plt.show()