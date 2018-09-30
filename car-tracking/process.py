import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def refine(image):
    # -1: must be 0; 1: must be 255; 0: either
    meta = [np.array([[1,1,1],[-1,0,0],[0,-1,0]]),
            np.array([[1,1,1],[0,0,0],[0,-1,-1]])]
    templates = []
    for t in meta:
        for i in range(4):
            templates.append(np.rot90(t, i))
            templates.append(np.rot90(t[:,::-1], i))
    templates = np.stack(templates, axis=2)
    h, w = image.shape
    count = 0
    for i in range(1, h-1):
        for j in range(1, w-1):
            if image[i][j] == 255: continue
            roi = image[i-1:i+2, j-1:j+2].astype(np.int32)
            if np.sum(roi, axis=(0,1)) == 0: continue
            if np.min(np.sum((roi-1).reshape((3,3,-1)) * templates < 0, axis=(0,1))) == 0:
                image[i][j] = 255
                count += 1
    print(count)
    # https://www.cnblogs.com/mikewolf2002/p/3327183.html


plt.ion()
address = 'http://admin:9092@10.87.187.225:8081/video'
cap = cv.VideoCapture(address)
print(cap.isOpened())

for t in range(200):
    cap = cv.VideoCapture(address)
    success, frame = cap.read(0)
    edges = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #edges = cv.GaussianBlur(edges, (7,7), 1.5, 1.5)
    #edges = cv.Canny(edges, 0, 30, 3)
    _, edges = cv.threshold(edges, 100, 255, cv.THRESH_BINARY)
    
    while True:
        refine(edges)
        plt.cla()
        plt.imshow(np.concatenate((frame[:,:,::-1], np.stack((edges,)*3, axis=2)), axis=1))
        plt.pause(0.003)


#def onMouse(e, x, y):
#    if e == cv.CV_EVENT_LBUTTONDOWN:
#        positions.append((x, y))
#        cv.circle(frame, cv.Point(x, y), 2, cv.Scalar(0, 0, 255), -1, cv.AA)
#

#        
#frame = cap.read()
#cv.imshow("capture", frame)
#cv.setMouseCallback("frame", onMouse)
#while len(positions) < 4:
#    pass
#cv.warpPerspective(frame, frame, positions)
#cv.imshow("capture", frame)
#cv.waitKey()
#cv.cvtColor(frame, frame, cv.CV_BGR2GRAY)
#cv.imshow("capture", frame)
#cv.waitKey()
#cv.threshold(frame, frame, 200, 255, cv.CV_THRESH_BINARY)
#cv.imshow("capture", frame)
#cv.waitKey()
#refine(frame)
#
#while True:
#    frame = cap.read()
#    cv.imshow("capture", frame)
#    if cv.waitKey(30) >= 0:
#        break
