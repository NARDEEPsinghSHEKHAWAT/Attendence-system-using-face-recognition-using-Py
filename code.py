import cv2
import numpy as np
import face_recognition as face_reg
import os
from datetime import datetime

def resize(img, size):
    width = int(img.shape[1] * size)
    height = int(img.shape[0] * size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)


path = 'students'
studentimg = []
studentname = []
mylist = os.listdir(path)
# print(mylist)

for cl in mylist:
    curimg = cv2.imread(f'{path}/{cl}')  # student img /guru.png/veda.png
    studentimg.append(curimg)
    studentname.append(os.path.splitext(cl)[0])


# print(studentname)
def findEncoding(images):
    encodelist = []
    for img in images:
        img = resize(img, 0.50)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeimg = face_reg.face_encodings(img)[0]
        encodelist.append(encodeimg)
    return encodelist


def MarkAttendence(name):
    with open('attendence.csv', 'r+') as f:
        mydatalist = f.readlines()
        Namelist = []
        for line in mydatalist:
            entry = line.split(',')
            Namelist.append(entry[0])
        if name not in Namelist:
            now = datetime.now()
            timestr = now.strftime('%H: %M')
            f.writelines(f'\n{name}, {timestr}')


encode_list = findEncoding(studentimg)

vid = cv2.VideoCapture(0)

while True:
    sucess, frame = vid.read()
    frames = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    #frames = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)

    faces_in_frame = face_reg.face_locations(frames)
    encode_in_frame = face_reg.face_encodings(frames, faces_in_frame)

    for encodeFace, faceloc in zip(encode_in_frame, faces_in_frame) :
        matches = face_reg.compare_faces(encode_list, encodeFace)
        facedis = face_reg.face_distance(encode_list, encodeFace)
        print(facedis)
        matchIndex = np.argmin(facedis)

        if matches[matchIndex] :
            name = studentname[matchIndex].upper()
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.rectangle(frame, (x1, y2-25), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            MarkAttendence(name)

        cv2.imshow('video', frame)
        cv2.waitKey(1)
