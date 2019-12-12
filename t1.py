import cv2
import os
from os import listdir
from os.path import isfile, join
import numpy

def facecrop(image):
    ### load haar cascade model ###
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    ### load an image ###
    img = cv2.imread(image)

    ### make a grayscale image from the loaded image ###
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    minisize = (img.shape[1],img.shape[0])
    miniframe = cv2.resize(img, minisize)

    ### find the face position by using the model from grayscale image ###
    ### scaleFactor = 1.3 ### minNeighbors = 2 ### minSize = 100x100 ###
    faces = cascade.detectMultiScale(gray,1.3, 2,minSize = (100,100))
    if (faces is None):
        print('Failed to detect face')
        return

    for f in faces:
        x, y, w, h = [ v for v in f ]
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))

        sub_face = img[y:y+h, x:x+w]
        fname, ext = os.path.splitext(image)
        cv2.imwrite(fname+"_cropped_"+ext, sub_face)


    return

# facecrop("1.jpg")
mypath = './jpg/test'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
images = numpy.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
  images[n] = cv2.imread( join(mypath,onlyfiles[n]) )

x = 0
for i in onlyfiles:
    print("croping " + i)
    facecrop(mypath+'/'+i)
    x = x+1
print(x)
print(len(onlyfiles))