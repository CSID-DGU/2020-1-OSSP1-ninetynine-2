import face_recognition
import cv2
import os
import argparse
import face_recognition
import numpy as np
import demo_texture
import tensorflow as tf

from face_detection import select_face
from face_swap import face_swap

from api import PRN
from utils.render import render_texture

import numpy as np
import os
from glob import glob
import scipy.io as sio
from skimage.io import imread, imsave
from skimage.transform import rescale, resize
from time import time
import argparse
import ast
import matplotlib.pyplot as plt
import argparse

from api import PRN
from utils.render import render_texture
#from pytorch_stylegan_encoder import encode_image as ei
import cv2
import dlib

from PRNet_master import demo_texture as dt



def best_value(face_encodings, known_face_encodings):
    A = []
    for face_encoding in face_encodings:
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        A.append(face_distances)
    return min(A)


known_finder_dir = "./known/"

file_list = os.listdir(known_finder_dir)
file_list_dir = [file for file in file_list if file.endswith("png")]
print(file_list_dir)

minju_image = face_recognition.load_image_file(known_finder_dir + file_list_dir[0])
minju_face_encoding = face_recognition.face_encodings(minju_image)[0]


known_face_encodings = [
    minju_face_encoding,
]
known_face_names = [
    "Min-Ju"
]

unknown_finder_dir = "./unknown/"
file_list2 = os.listdir(unknown_finder_dir)
file_list_dir2 = [file for file in file_list2 if file.startswith("unknown")]
unknown_dir = unknown_finder_dir+file_list_dir2[0]

print("this?")
print(file_list_dir2)
print(unknown_dir)

after_face_dir = "./npysave/"

after_face_list = os.listdir(after_face_dir)
after_face_list_dir = [file for file in after_face_list if file.endswith("stylegan")]

print(after_face_list_dir)
after_file_name = "/000_000.jpg"

after_face_list_full_dir = []
for i in range(len(after_face_list_dir)):
    after_face_list_full_dir.append(after_face_dir + after_face_list_dir[i] + after_file_name)
print(after_face_list_full_dir)



"""

raise Exception

"""
# Load an image with an unknown face
unknown_image = face_recognition.load_image_file(unknown_dir)
face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

best = best_value(face_encodings, known_face_encodings)

print(face_locations)

total_face_count = len(face_locations)
before_file_path = "./npysave/"

file_list = os.listdir(before_file_path)
file_list_dir = [file for file in file_list if file.endswith("stylegan")]
print(file_list_dir)

#0 < x < total_face_count
full_dir_name = []
for i in range(len(file_list_dir)):
    file_list = os.listdir(before_file_path + str(file_list_dir[i]) + "/")
    full_dir_name.append([file for file in file_list if file.endswith(".jpg")])

print(full_dir_name)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")

## 비디오 읽어오기

cam = unknown_image
color_green = (0,255,0)
line_width = 3
rgb_image = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
rects = detector(rgb_image, 1)
print(rects)
print(rects[0].left())




# 최고 닮은 도가 임계 값을 넘지 못하면 ㅂ2
if best > 0.5:
    best = 0

find_my = 0

for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Unknown"
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    print(face_distances)
    #하나만 찾는데에는 쓸데없음
    best_match_index = np.argmin(face_distances)
    if face_distances == best:
        name = known_face_names[best_match_index]
        find_my = (top, right, bottom, left)
#445 1619 534 1529
#unknown_image
print(best_value(face_encodings, known_face_encodings))
print(find_my)




image = face_recognition.load_image_file(unknown_dir)
face_locations = face_recognition.face_locations(image)
print(face_locations)
img = cv2.imread(unknown_dir)
img2 = cv2.imread(unknown_dir)



#! /usr/bin/env python
count_name = 0


for i in range(len(face_locations)):
    tf.reset_default_graph()

    file_list = os.listdir("./data/")
    print(file_list)

    name_list = []
    for name in file_list:
        name_list.append(name)

####
#    args_src = "./data/" + name_list[count_name]
    args_src = after_face_list_full_dir[i]

    args_dst = "./unknown/unknown.jpg"
    args_out = "./output/result"

    print("count_name",count_name)
    print("face_locations",len(face_locations))
    print("face_locations[count_name]",len(face_locations[count_name]))

    #print(rects[0].left())
    Y = int((rects[count_name].bottom()-rects[count_name].top())/2)
    X = int((rects[count_name].right()-rects[count_name].left())/2)

    # Read images
    src_img = cv2.imread(after_face_list_full_dir[i])
    #rects = 랜드마크 뽑은 이미지
    #src_img = 원본
    dst_img = img[rects[count_name].top()-Y:rects[count_name].bottom()+Y,rects[count_name].left()-X:rects[count_name].right()+X]
    return_img = dt.main(dst_img, src_img, "")
    return_img = cv2.cvtColor(return_img,cv2.COLOR_RGB2BGR)
    img[rects[count_name].top()-Y:rects[count_name].bottom()+Y,rects[count_name].left()-X:rects[count_name].right()+X] = return_img


#    cv2.imwrite("asdf.jpg" , img)
    count_name += 1

img[int(find_my[0]*0.7):int(find_my[2]*1.3),int(find_my[3]*0.7):int(find_my[1]*1.3)] = img2[int(find_my[0]*0.7):int(find_my[2]*1.3),int(find_my[3]*0.7):int(find_my[1]*1.3)]

cv2.imwrite("./result/output.jpg" , img)
cv2.destroyAllWindows()
