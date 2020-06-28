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


def best_value(face_encodings, known_face_encodings):
    A = []
    for face_encoding in face_encodings:
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        A.append(face_distances)
    return min(A)


minju_image = face_recognition.load_image_file("./known/minwoo.jpg")
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


for i in range(4):

    count_name = 2
    parser = argparse.ArgumentParser(description='FaceSwapApp')
    parser.add_argument('--src',  help='Path for source image')
    parser.add_argument('--dst',  help='Path for target image')
    parser.add_argument('--out',  help='Path for storing output images')
    parser.add_argument('--warp_2d', default=False, action='store_true', help='2d or 3d warp')
    parser.add_argument('--correct_color', default=False, action='store_true', help='Correct color')
    parser.add_argument('--no_debug_window', default=False, action='store_true', help='Don\'t show debug window')
    args = parser.parse_args()

    file_list = os.listdir("./data/")
    print(file_list)

    name_list = []
    for name in file_list:
        name_list.append(name)

    args_src = "./data/" + name_list[count_name]
    args_dst = "./unknown/unknown.jpg"
    args_out = "./output/result"

    Y = int((face_locations[count_name][2]-face_locations[count_name][0])/2)
    X = int((face_locations[count_name][1]-face_locations[count_name][3])/2)


    # Read images
    src_img = cv2.imread(args_src)
#    dst_img = cv2.imread(args.dst)
    dst_img = img[face_locations[count_name][0]-Y:face_locations[count_name][2]+Y,face_locations[count_name][3]-X:face_locations[count_name][1]+X]
    # Select src face
    src_points, src_shape, src_face = select_face(src_img)
    # Select dst face
    dst_points, dst_shape, dst_face = select_face(dst_img)

    if src_points is None or dst_points is None:
        print('Detect 0 Face !!!')
        exit(-1)

    os.environ['CUDA_VISIBLE_DEVICES'] = "0" # GPU number, -1 for CPU
    prn = PRN(is_dlib = True)

    demo_texture.texture_editing(prn, "./unknown/unknown.jpg", src_img, "./unknown/unknown.jpg", mode = 1)
#    img[face_locations[count_name][0]-Y:face_locations[count_name][2]+Y,face_locations[count_name][3]-X:face_locations[count_name][1]+X] = face_swap(src_face, dst_face, src_points, dst_points, dst_shape, dst_img, args)
    cv2.imwrite("asdf.jpg" , img)
    count_name += 1


img[find_my[0]:find_my[2],find_my[3]:find_my[1]] = img2[find_my[0]:find_my[2],find_my[3]:find_my[1]]
cv2.imwrite("0614ttt.jpg" , img)
cv2.destroyAllWindows()
