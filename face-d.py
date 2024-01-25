import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

video_capture = cv2.VideoCapture(0)

messi_image = face_recognition.load_image_file("images/messi.jpeg")
messi_encoding = face_recognition.face_encodings(messi_image)[0]

ronaldo_image = face_recognition.load_image_file("images/ronaldo.jpeg")
ronaldo_encoding = face_recognition.face_encodings(ronaldo_image)[0]

neymar_image = face_recognition.load_image_file("images/neymar.jpeg")
neymar_encoding = face_recognition.face_encodings(neymar_image)[0]

gareth_image = face_recognition.load_image_file("images/gareth.jpeg")
gareth_encoding = face_recognition.face_encodings(gareth_image)[0] 

halaand_image = face_recognition.load_image_file("images/halaand.jpeg")
halaand_encoding = face_recognition.face_encodings(halaand_image)[0]

known_face_encoding = [
    messi_encoding,
    ronaldo_encoding,
    neymar_encoding,
    gareth_encoding,
    halaand_encoding
]

known_face_names = [
    "Messi",
    "Ronaldo",
    "Neymar",
    "Gareth",
    "Halaand"
]

student = known_face_names.copy()

face_locations = []
face_encodings = []
face_names = []
s = True 


now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date + ".csv","w+",newline='')
lnwriter = csv.writer(f)

while True :
    _,frame = video_capture.read()
    small_frame =cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    # rgb_small_frame = small_frame[:,:,::-1]
    if s:
        face_locations =face_recognition.face_locations(small_frame)
        face_encodings =face_recognition.face_encodings(small_frame,face_locations)
        face_names =[]
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
            name = ""
            face_distance =face_recognition.face_distance(known_face_encoding,face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                
            face_names.append(name)
            if name in known_face_names:
                if name in student:
                    student.remove(name)
                    print(student)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name,current_time])
            
            
    cv2.imshow("Attendence System",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
video_capture.release()
cv2.destroyAllWindows()
f.close()