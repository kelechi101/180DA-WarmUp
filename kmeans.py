"""
References: 
find dominant colors on an image:https://code.likeagirl.io/finding-dominant-colour-on-an-image-b4e075f98097
kmeans:https://github.com/opencv/opencv/blob/master/samples/python/kmeans.py
BoundingBox: https://answers.opencv.org/question/200861/drawing-a-rectangle-around-a-color-as-shown/?fbclid=IwAR3_P7pmzYS7Mw8sSBtK5d7szhg-RNjCkZzlrln65rmqrbxqHm10JTxzG1I%22%22%22   
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist
def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar
cap = cv.VideoCapture('blue.MOV') #import video 
while(1):

    _, frame = cap.read()
    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([100,55,55])
    upper_blue = np.array([105,255,255])
    # Threshold the HSV image to get only blue colors
    mask = cv.inRange (frame, lower_blue, upper_blue)
    bluecnts = cv.findContours(mask.copy(),
                              cv.RETR_EXTERNAL,
                              cv.CHAIN_APPROX_SIMPLE)[-2]
    if len(bluecnts)>0:
        blue_area = max(bluecnts, key=cv.contourArea)
        (xg,yg,wg,hg) = cv.boundingRect(blue_area)
        cv.rectangle(frame,(xg,yg),(xg+wg, yg+hg),(0,255,0),2)
        

        cv.imshow('frame',frame)
        img = frame[yg:yg+hg,xg:xg+wg]
        img=cv.cvtColor(img,cv.COLOR_BGR2RGB)

        img = img.reshape((img.shape[0] * img.shape[1],3)) #represent as row*column,channel number
        clt = KMeans(n_clusters=3) #cluster number
        clt.fit(img)

        hist = find_histogram(clt)
        bar = plot_colors2(hist, clt.cluster_centers_)

        plt.axis("off")
        plt.imshow(bar)
        plt.show()


    
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break