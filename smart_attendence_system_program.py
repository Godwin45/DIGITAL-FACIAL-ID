import cv2
import numpy as np
import face_recognition as face_rec
import os
import pyttsx3 as textSpeach
from datetime import  datetime
import mediapipe as mp

engine = textSpeach.init()
engine = textSpeach.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('rate', 190)
engine.setProperty('volume', 3.0)
engine.setProperty('voice', voices[1].id)

def resize(img, size) :
    width = int(img.shape[1]*size)
    height = int(img.shape[0] * size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation= cv2.INTER_AREA)

path = 'student_images'
studentImg = []
studentName = []
myList = os.listdir(path)
for cl in myList :
    curimg = cv2.imread(f'{path}/{cl}')
    studentImg.append(curimg)
    studentName.append(os.path.splitext(cl)[0])

def findEncoding(images) :
    imgEncodings = []
    for img in images :
        img = resize(img, 0.50)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeimg = face_rec.face_encodings(img)[0]
        imgEncodings.append(encodeimg)
    return imgEncodings
def MarkAttendence(name):
    with open('attendence.csv', 'r+') as f:
        myDatalist =  f.readlines()
        nameList = []
        for line in myDatalist :
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            now = datetime.now()
            timestr = now.strftime('%H:%M')
            f.writelines(f'\n{name}, {timestr}')
            statment = str('Welcome to school   ' + name)
            engine.say(statment)
            engine.runAndWait()

        




EncodeList = findEncoding(studentImg)

vid = cv2.VideoCapture(0)

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces = 2)
drawSpec = mpDraw.DrawingSpec(thickness = 1, circle_radius = 2)
drawSpec1 = mpDraw.DrawingSpec(color=(0, 0, 255), thickness = 1, circle_radius = 2)

while True :
    success, frame = vid.read()
    Smaller_frames = cv2.resize(frame, (0,0), None, 0.25, 0.25)

    facesInFrame = face_rec.face_locations(Smaller_frames)
    encodeFacesInFrame = face_rec.face_encodings(Smaller_frames, facesInFrame)

    for encodeFace, faceloc in zip(encodeFacesInFrame, facesInFrame) :
        matches = face_rec.compare_faces(EncodeList, encodeFace)
        facedis = face_rec.face_distance(EncodeList, encodeFace)
        # print(facedis)
        matchIndex = np.argmin(facedis)

        if matches[matchIndex] :
            name = studentName[matchIndex].upper()
            name1 = "NOT STUDENT"
            imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = faceMesh.process(imgRGB)
            if results.multi_face_landmarks:
                for faceLms in results.multi_face_landmarks:
                    
                    mpDraw.draw_landmarks(frame, faceLms, mpFaceMesh.FACE_CONNECTIONS,
                    drawSpec, drawSpec)

                    for id, lm in enumerate(faceLms.landmark):
                        ih, iw, ic, = frame.shape
                        x, y = int(lm.x*iw), int(lm.y*ih)
                    

                        cv2.putText(frame, name, (20, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                        MarkAttendence(name)
        else:
             mpDraw.draw_landmarks(frame, faceLms, mpFaceMesh.FACE_CONNECTIONS,
                        drawSpec1, drawSpec1)

             for id, lm in enumerate(faceLms.landmark):
                        ih, iw, ic, = frame.shape
                        x, y = int(lm.x*iw), int(lm.y*ih)
                    

                        cv2.putText(frame, name1, (20, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                        MarkAttendence(name)
                                    
                        # statment1 = str('SORRY, YOU ARE NOT A STUDENT   ')
                        # engine.say(statment1)
                        # engine.runAndWait()

                        
            
        

    cv2.imshow('video',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
