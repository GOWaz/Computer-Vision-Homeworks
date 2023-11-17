import cv2 as cv
import numpy as np

img = cv.imread('C:/Users/tonyb/Desktop/Open CV/HomeWorks/Computer-Vision-Homeworks/HM4/assets/y.jpg')
temp = cv.imread('C:/Users/tonyb/Desktop/Open CV/HomeWorks/Computer-Vision-Homeworks/HM4/assets/yy.jpg')

# img = cv.resize(img,(600,600))
# temp = cv.resize(temp,(600,600))

cv.imshow('img1',img)
cv.waitKey(0)

cv.imshow('temp',temp)
cv.waitKey(0)


result = cv.matchTemplate(img, temp, cv.TM_CCORR_NORMED)
cv.imshow('result', result)
cv.waitKey(0)

min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

print(max_loc)
print(max_val)

w = img.shape[1]
h = img.shape[0]

cv.rectangle(img,max_loc,(max_loc[0] +400, max_loc[1]+600),(0,255,255),2)
# img = cv.resize(img,(600,600))
cv.imshow('new result',img)
cv.waitKey(0)

threshold = .10
yloc, xloc = np.where(result >= threshold)
print(len(xloc))

for (x,y) in zip(xloc, yloc):
    cv.rectangle(img,(x,y),(x+400,y+600),(0,255,255),2)



rectangles = []
for (x,y) in zip(xloc, yloc):
    rectangles.append([int(x), int(y), int(w), int(h)])
    rectangles.append([int(x), int(y), int(w), int(h)])

rectangles, weights = cv.groupRectangles(rectangles, 1,0.2)

for (x,y, w, h) in rectangles :
    cv.rectangle(img,(x,y),(x+w, y+h),(0,255,255),2)

cv.imshow('result',img)    
cv.waitKey(0)