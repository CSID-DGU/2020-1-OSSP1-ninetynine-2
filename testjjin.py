import face_recognition
import cv2
import os
import argparse
import face_recognition
import numpy as np
import demo_texture

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
import cv2

import random

def best_value(face_encodings, known_face_encodings):
    A = []
    for face_encoding in face_encodings:
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        A.append(face_distances)
    return min(A)

def texture_editing(prn, image_path, ref_path, output_path, mode = 1):
    # read image
    image = cv2.imread(image_path,cv2.IMREAD_COLOR)
    #[h, w, _] = image.shape
    #h=int(h*0.5)
    #w=int(w*0.5)
    #image = cv2.resize(image ,(w,h),interpolation=cv2.INTER_AREA)
    [h, w, _] = image.shape
    print("h_w_",h,w)

    #-- 1. 3d reconstruction -> get texture.
    pos = prn.process(image)
    vertices = prn.get_vertices(pos)
    image = image/255.
    texture = cv2.remap(image, pos[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))

    #-- 2. Texture Editing
    Mode = mode
    # change part of texture(for data augumentation/selfie editing. Here modify eyes for example)

    # change whole face(face swap)
    if Mode == 1:
        # texture from another image or a processed texture
#        ref_image = imread(ref_path)
        ref_image = ref_path
        ref_pos = prn.process(ref_image)
        ref_image = ref_image/255.
        ref_texture = cv2.remap(ref_image, ref_pos[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
        ref_vertices = prn.get_vertices(ref_pos)
        new_texture = ref_texture#(texture + ref_texture)/2.

    else:
        print('Wrong Mode! Mode should be 0 or 1.')
        exit()


    #-- 3. remap to input image.(render)
    vis_colors = np.ones((vertices.shape[0], 1))
    face_mask = render_texture(vertices.T, vis_colors.T, prn.triangles.T, h, w, c = 1)
    face_mask = np.squeeze(face_mask > 0).astype(np.float32)

    new_colors = prn.get_colors_from_texture(new_texture)
    new_image = render_texture(vertices.T, new_colors.T, prn.triangles.T, h, w, c = 3)
    new_image = image*(1 - face_mask[:,:,np.newaxis]) + new_image*face_mask[:,:,np.newaxis]

    # Possion Editing for blending image
    vis_ind = np.argwhere(face_mask>0)
    vis_min = np.min(vis_ind, 0)
    vis_max = np.max(vis_ind, 0)
    center = (int((vis_min[1] + vis_max[1])/2+0.5), int((vis_min[0] + vis_max[0])/2+0.5))
    output = cv2.seamlessClone((new_image*255).astype(np.uint8), (image*255).astype(np.uint8), (face_mask*255).astype(np.uint8), center, cv2.NORMAL_CLONE)

    # save output
    imsave(output_path, output)
    print('Done.')


minju_image = face_recognition.load_image_file("./known/jimin.jpg")
minju_face_encoding = face_recognition.face_encodings(minju_image)[0]


known_face_encodings = [
    minju_face_encoding,
]
known_face_names = [
    "Ji-Min"
]
# Load an image with an unknown face

unknown_image = face_recognition.load_image_file("./unknown/unknown.jpg")
#unkonwn_HSV =cv2.cvtColor(unknown_image,cv2.COLOR_BGR2HSV)
#height, width, channel = unknown_image.shape
#v = 0

#for i in range(height):
#    for j in range(width):
#        v+=unkonwn_HSV[i][j][2]
#v = int( v / (height*width))

#b, g, r = cv2.split(unknown_image)

#blur_b=cv2.GaussianBlur(b,(5,5),3)
#blur_g=cv2.GaussianBlur(g,(5,5),3)
#blur_r=cv2.GaussianBlur(r,(5,5),3)

#sub_b=cv2.subtract(b,blur_b)
#sub_g=cv2.subtract(g,blur_g)
#sub_r=cv2.subtract(r,blur_r)

#add_b=cv2.add(b,sub_b)
#add_g=cv2.add(g,sub_g)
#add_r=cv2.add(r,sub_r)

#b=cv2.equalizeHist(add_b)
#g=cv2.equalizeHist(add_g)
#r=cv2.equalizeHist(add_r)

#if(v < 100):
#    b=cv2.add(b,10)
#    g=cv2.add(g,10)
#    r=cv2.add(r,10)

#unknown_image=cv2.merge((b,g,r))
#unknown_image=cv2.fastNlMeansDenoisingColored(unknown_image)

#cnn 개느림
face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
best = best_value(face_encodings, known_face_encodings)

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

image = face_recognition.load_image_file("./unknown/unknown.jpg")
face_locations = face_recognition.face_locations(image)
print(face_locations)
img = cv2.imread("./unknown/unknown.jpg")
img2 = cv2.imread("./unknown/unknown.jpg")
print(face_locations[0][0])
print(face_locations[0][1])
print(face_locations[0][2])
print(face_locations[0][3])



"""
cv2.imshow("asdf", img[0:500, 100: 400])

cv2.waitKey(0)

cv2.imshow("asdf", img[face_locations[0][0]:face_locations[0][2],face_locations[0][3]:face_locations[0][1]])

cv2.waitKey(0)

cv2.destroyAllWindows()
"""
#! /usr/bin/env python
# 무작위로 생성 이미지 파일 선정 face swap 진행
GENERATOR_IMAGE_FILE_NUMBER = 4
count_name = 0
list_random = []

ran_num =random.randint(0,GENERATOR_IMAGE_FILE_NUMBER-1)
for i in range(GENERATOR_IMAGE_FILE_NUMBER):
    while ran_num in list_random:
        ran_num = random.randint(0,GENERATOR_IMAGE_FILE_NUMBER-1)
    list_random.append(ran_num)

for i in range(len(face_locations)):
    print(count_name)
    file_list = os.listdir("./data/")

    print(file_list)

    name_list = []
    for name in file_list:
        name_list.append(name)

    args_src = "./data/" + name_list[list_random[count_name]]
    args_dst = "./unknown/unknown.jpg"
    args_out = "./output/result"

    print("list_random",len(list_random))
    print("face_locations",len(face_locations))
    print("list_random[count_name]",list_random[count_name])
    print("face_locations",len(face_locations[list_random[count_name]]))
    Y = int((face_locations[list_random[count_name]][2]-face_locations[list_random[count_name]][0])/2)
    X = int((face_locations[list_random[count_name]][1]-face_locations[list_random[count_name]][3])/2)

    # Read images
    src_img = cv2.imread(args_src)
    #dst_img = cv2.imread(args.dst)

    dst_img = img[face_locations[list_random[count_name]][0]-Y:face_locations[list_random[count_name]][2]+Y,face_locations[list_random[count_name]][3]-X:face_locations[list_random[count_name]][1]+X]
    # Select src face

    os.environ['CUDA_VISIBLE_DEVICES'] = "0" # GPU number, -1 for CPU
    prn = PRN(is_dlib = True)

    #texture_editing(prn, "./unknown/unknown.jpg", src_img, "./unknown/unknown.jpg", mode = 1)

    cv2.imwrite("asdf.jpg" , img)
    count_name += 1
print("for 통과 2")
"""

    if src_points is None or dst_points is None:
        print('Detect 0 Face !!!')
        exit(-1)

"""

img[find_my[0]:find_my[2],find_my[3]:find_my[1]] = img2[find_my[0]:find_my[2],find_my[3]:find_my[1]]
cv2.imwrite("AAAAAA.jpg" , img)
cv2.destroyAllWindows()
