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


def best_value(face_encodings, known_face_encodings):
    A = []
    for face_encoding in face_encodings:
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        A.append(face_distances)
    return min(A)

def texture_editing(prn, image_path, ref_path, output_path, mode = 1):
    # read image
    image = imread(image_path)
    [h, w, _] = image.shape
    if h > 2000 or w > 2000:
        h = int(h*0.6)
        w = int(w*0.6)
        image = cv2.resize(image ,(w,h), interpolation=cv2.INTER_AREA)

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
    "Min-Ju"
]

# Load an image with an unknown face
unknown_image = face_recognition.load_image_file("./unknown/unknown.jpg")
#cnn 개느림
face_locations = face_recognition.face_locations(unknown_image)

face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

best = best_value(face_encodings, known_face_encodings)

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
count_name = 0


for i in range(len(face_locations)):

    file_list = os.listdir("./data/")
    print(file_list)

    name_list = []
    for name in file_list:
        name_list.append(name)

    args_src = "./data/" + name_list[count_name]
    args_dst = "./unknown/unknown.jpg"
    args_out = "./output/result"

    print("count_name",count_name)
    print("face_locations",len(face_locations))
    print("face_locations[count_name]",len(face_locations[count_name]))

    Y = int((face_locations[count_name][2]-face_locations[count_name][0])/2)
    X = int((face_locations[count_name][1]-face_locations[count_name][3])/2)

    # Read images
    src_img = cv2.imread(args_src)
    #dst_img = cv2.imread(args.dst)

    src_img =cv2.cvtColor(src_img,cv2.COLOR_BGR2RGB)
    dst_img = img[face_locations[count_name][0]-Y:face_locations[count_name][2]+Y,face_locations[count_name][3]-X:face_locations[count_name][1]+X]
    # Select src face
    dst_img =cv2.cvtColor(dst_img,cv2.COLOR_BGR2RGB)
    os.environ['CUDA_VISIBLE_DEVICES'] = "0" # GPU number, -1 for CPU
    tf.reset_default_graph ()
    prn = PRN(is_dlib = True)

    texture_editing(prn, "./unknown/unknown.jpg", src_img, "./unknown/unknown.jpg", mode = 1)

    img =cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    cv2.imwrite("asdf.jpg" , img)
    count_name += 1

"""

    if src_points is None or dst_points is None:
        print('Detect 0 Face !!!')
        exit(-1)

"""

img[find_my[0]:find_my[2],find_my[3]:find_my[1]] = img2[find_my[0]:find_my[2],find_my[3]:find_my[1]]

cv2.imwrite("AAAAAB.jpg" , img)
cv2.destroyAllWindows()
