import cv2
from numpy import *
import face_recognition
import os
from datetime import datetime

import numpy

path = 'images'
images = []
personName = []
myList = os.listdir(path)
print(myList)

#now we have to seprate the name from the images, so that we can use the name in the camera's frame

#for reading are images from the list we will use cv2 module

for cu_img in myList:
    current_Img = cv2.imread(f'{path}/{cu_img}')
    
    #To add images into images list, we will use append function and for adding name also we use append function and store the name in personName list with os.path.splitext(cu_img)[0]
    
    #now question arrieses that how it get spilt, so here name is priyanka.jpg so here priyanka is zeroth component and jpg is the first component and we want zeroth component so will take [0]component 
    
    images.append(current_Img)
    personName.append(os.path.splitext(cu_img)[0])
print(personName)

#Now we will generate face encode so that to make a generalize function bcoz we can have 'n' numbers of images so, if we make a generalize function we do not need to do repetatively encoding for the images

#here we will pass the images as we have to encode that the person in front of camera do mataches with the image saved in images list or not if do then the name of the person display automatically as it is saved in the code but if not then the 'unknown' as name of person will display in the camera 

#Here Encoding means here our face_recongition module is based on dlib, and what the dlib do, it encodes our face on 128 different features, on the basis of which a machine reconginise the difference between two different faces(basically comparision)

def faceEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# By using we can find the different 128 features code number, by this differnet number two faces are differentiated, and algorithms used in this is "hog" algorithm "print(faceEncodings(images))"

encodeListKnown = faceEncodings(images)      
print("ALL Encodeings Complete!!!!")

def attendance(name):
    with open('attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        
        if name not in nameList:
            time_now = datetime.now()
            tStr = time_now.strftime('%H:%M:%S')
            dStr = time_now.strftime('%d/%m/%Y')
            f.writelines(f'{name},{tStr},{dStr}\n')
            
# Now we will read our Camera VideoCapture feature of cv2 module, Now we have to give the id to the camera, if we are using laptop camera then the id is 0, and if we are using external webcam then the id will be 1
cap = cv2.VideoCapture(0)

# For read camera frame we will call read method through cap object, and it will return two things, i.e. ret, frame. now we will resize it, bcoz if anyone have high resolution in there camera to correct that, by resize function in dsize

while True:
    ret, frame = cap.read()
    faces = cv2.resize(frame, (0,0), None, 0.25, 0.25)
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)
    
    #Now finding face location with face_locations function and will find face ecodings 
    
    facesCurrentFrame = face_recognition.face_locations(faces)
    encodesCurrentFrame = face_recognition.face_encodings(faces, facesCurrentFrame)
    
    #for passing both parameter simultaneously we use zip function 
    
    #Now we have to do face comparision and to find face distances
    
    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        
        matchIndex = numpy.argmin(faceDis)
        
        #Now matching if the matching index is minimum then is matches, by using matchIndex variable if found then add the person name into personName  list, if not, then not matches
        
        if matches[matchIndex]:
            name = personName[matchIndex].upper()
           # print(name)
            
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4, x2*4, y2*4, x1*4 
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.rectangle(frame, (x1,y2-35), (x2,y2),(0,255,0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            attendance(name)
            
    cv2.imshow("Camera", frame)
    if cv2.waitKey(10) == 13:
        break
    
cap.release()
cv2.destroyAllWindows()


