import sys
import os
import dlib
import glob
from skimage import io
import numpy as np
import cv2

img = cv2.imread("orig.png") # 가상 인물 이미지
img2 = cv2.imread("input1.png") # 원본 인물 이미지

height, width, channel = img.shape
height2, width2, channel2 = img2.shape # 사진 크기 정함

predictor_path = 'shape_predictor_81_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

landmarks = []
landmarks_2 = [] 

bott_line=[] #관자부터 턱선 라인 
up_line=[] # 이마 라인

bott_line_2 = []
up_line_2 = []

max_x = 0
max_x_2 = 0 
min_x = 9999
min_x_2 = 9999

bott = [0 for i in range(width)]
up = [999 for i in range(width)] #최대 X 값 만큼 배열의 크기 할당

bott2 = [0 for i in range(width2)]
up2 = [999 for i in range(width2)] #최대 X 값 만큼 배열의 크기 할당

faces = detector(img,1)
faces_2 = detector(img2 ,1)
def new_up_line_decetector(c):
    a = 0
    b = 0 
    global landmarks,landmarks_2,up,up2,max_x,min_x,max_x_2,min_x_2
    if c==1:# c==1은 img에 대한 상황
        tu1 = (landmarks[77][0],landmarks[77][1])
        tu2 = (landmarks[75][0],landmarks[75][1])
        tu3 = (landmarks[76][0],landmarks[76][1])
        tu4 = (landmarks[68][0],landmarks[68][1])
        tu5 = (landmarks[69][0],landmarks[69][1])
        tu6 = (landmarks[70][0],landmarks[70][1])
        tu7 = (landmarks[71][0],landmarks[71][1])
        tu8 = (landmarks[80][0],landmarks[80][1])
        tu9 = (landmarks[72][0],landmarks[72][1])
        tu10 = (landmarks[73][0],landmarks[73][1])
        tu11 = (landmarks[79][0],landmarks[79][1])
        tu12 = (landmarks[74][0],landmarks[74][1])
        tu13 = (landmarks[78][0],landmarks[78][1])
        tu_list=[tu1,tu2,tu3,tu4,tu5,tu6,tu7,tu8,tu9,tu10,tu11,tu12,tu13]
        for i in range(68,80):
            x1 = tu_list[i-68][0]
            y1 = tu_list[i-68][1]
            x2 = tu_list[i-67][0]
            y2 = tu_list[i-67][1]
            max_x = max(max_x,x1)
            max_x = max(max_x,x2)
            if(min_x != 0):
                min_x = min(min_x,x1)
                min_x = min(min_x,x2)
            else:
                min_x = min(x1,x2)    
            if x1 - x2 !=0:
                a = (y1-y2) / (x1-x2)
                b = y1 - a*x1
                for j in range(x1,x2):
                    up[j] = min(min(y1,y2),up[j]) 
            elif x1==x2:
                up[x1] = min(min(y1,y2),up[x1])  
                bott[x1] = max(y1,y2)           
    elif c==2:# c==2는 img2에 대한 상황
        tu1 = (landmarks_2[77][0],landmarks_2[77][1])
        tu2 = (landmarks_2[75][0],landmarks_2[75][1])
        tu3 = (landmarks_2[76][0],landmarks_2[76][1])
        tu4 = (landmarks_2[68][0],landmarks_2[68][1])
        tu5 = (landmarks_2[69][0],landmarks_2[69][1])
        tu6 = (landmarks_2[70][0],landmarks_2[70][1])
        tu7 = (landmarks_2[71][0],landmarks_2[71][1])
        tu8 = (landmarks_2[80][0],landmarks_2[80][1])
        tu9 = (landmarks_2[72][0],landmarks_2[72][1])
        tu10 = (landmarks_2[73][0],landmarks_2[73][1])
        tu11 = (landmarks_2[79][0],landmarks_2[79][1])
        tu12 = (landmarks_2[74][0],landmarks_2[74][1])
        tu13 = (landmarks_2[78][0],landmarks_2[78][1])
        tu_list=[tu1,tu2,tu3,tu4,tu5,tu6,tu7,tu8,tu9,tu10,tu11,tu12,tu13]
        for i in range(68,80):
            x1 = tu_list[i-68][0]
            y1 = tu_list[i-68][1]
            x2 = tu_list[i-67][0]
            y2 = tu_list[i-67][1]
            max_x_2 = max(max_x,x1)
            max_x_2 = max(max_x,x2)
            if(min_x_2 != 0):
                min_x_2 = min(min_x,x1)
                min_x_2 = min(min_x,x2)
            else:
                min_x_2 = min(x1,x2)  
            if x1 - x2 !=0:
                a = (y1-y2) / (x1-x2)
                b = y1 - a*x1
                for j in range(x1,x2):
                    up2[j] = min(int(a*j+b) , up2[j])  
            elif x1==x2:
                up2[x1] = min(min(y1,y2),up2[x1]) 
                bott2[x1] = max(y1,y2)       

def bott_line_dectector(c):
    a = 0
    b = 0 
    global landmarks,landmarks_2,bott,bott2,max_x,max_x_2,min_x,min_x_2
    if c==1:# c==1은 img에 대한 상황 
        for i in range(0,16):
            x1 = landmarks[i][0]
            y1 = landmarks[i][1]
            x2 = landmarks[i+1][0]
            y2 = landmarks[i+1][1]
            max_x = max(max_x,x1)
            max_x = max(max_x,x2)
            if(min_x != 0):
                min_x = min(min_x,x1)
                min_x = min(min_x,x2)
            else:
                min_x = min(x1,x2)  
            if x1 - x2 !=0:
                a = (y1-y2) / (x1-x2)
                b = y1 - a*x1
                for j in range(x1,x2):
                    bott[j] =  max(int(a*j+b),bott[j])  
            elif x1==x2:
                up[x1] = min(min(y1,y2),up[x1]) 
                bott[x1] = max(y1,y2) 
    elif c==2:# c==2는 img2에 대한 상황
        for i in range(0,16):
            x1 = landmarks_2[i][0]
            y1 = landmarks_2[i][1]
            x2 = landmarks_2[i+1][0]
            y2 = landmarks_2[i+1][1]
            max_x_2 = max(max_x_2,x1)
            max_x_2 = max(max_x_2,x2)
            if(min_x_2 != 0):
                min_x_2 = min(min_x,x1)
                min_x_2 = min(min_x,x2)
            else:
                min_x_2 = min(x1,x2)  
            if x1 - x2 !=0:
                a = (y1-y2) / (x1-x2)
                b = y1 - a*x1
                for j in range(x1,x2):
                    bott2[j] = max(int(a*j+b),bott2[j]) 
            elif x1==x2:
                up2[x1] = min(min(y1,y2),up2[x1])  
                bott2[x1] = max(y1,y2)

for k, d in enumerate(faces):
    shape = predictor(img, d)
    for p in shape.parts():
        pot = (p.x,p.y)
        landmarks.append(pot)
    for num in range(shape.num_parts):
        cv2.circle(img, (shape.parts()[num].x, shape.parts()[num].y), 3, (0,255,0), -1)

new_up_line_decetector(1)
bott_line_dectector(1)    

x1 = landmarks[0][0]
x2 =landmarks[77][0]
y1 = landmarks[0][1]
y2 =landmarks[77][1]                

if x2 - x1 !=0:
    a = (y2-y1)/(x2-x1)
    if a >0:
            max_x = max(max_x,x1)
            max_x = max(max_x,x2)
            min_x = min(min_x,x1)
            min_x = min(min_x,x2)
            b = y1 - a*x1
            for j in range(x1,x2):
                up[j] = max(int(a*j+b) , up[j])          
    elif a<0:
            max_x = max(max_x_2,x1)
            max_x = max(max_x_2,x2)
            min_x = min (min_x_2,x1)
            min_x = min(min_x_2,x2)
            b = y1 - a*x1
            for j in range(x1,x2):
                bott[j] = int(a*j+b)  
else:
    up[x1] = max(max(y1,y2),up[x1])
    bott[x1] = min(y1,y2)     

for k, d in enumerate(faces_2):
    shape = predictor(img, d)
    for p in shape.parts():
        pot = (p.x,p.y)
        landmarks_2.append(pot)
    for num in range(shape.num_parts):
        cv2.circle(img2, (shape.parts()[num].x, shape.parts()[num].y), 3, (0,255,0), -1)

bott_line_dectector(2)
new_up_line_decetector(2) 

x1 = landmarks_2[0][0]
x2 = landmarks_2[77][0]
y1 = landmarks_2[0][1]
y2 = landmarks_2[77][1]                

if x2 - x1 !=0:
    a = (y2-y1)/(x2-x1)
    if a >0:
            max_x_2 = max(max_x,x1)
            max_x_2 = max(max_x,x2)
            min_x_2 = min(min_x,x1)
            min_x_2 = min(min_x,x2)
            b = y1 - a*x1
            for j in range(x1,x2):
                up2[j] = min(int(a*j+b) , up2[j])          
    elif a<0:
            max_x_2 = max(max_x_2,x1)
            max_x_2 = max(max_x_2,x2)
            min_x_2 = min (min_x_2,x1)
            min_x_2 =min(min_x_2,x2)
            b = y1 - a*x1
            for j in range(x1,x2):
                bott2[j] = max(int(a*j+b), bott2[j])  
else:
    up2[x1] = min(min(y1,y2),up2[x1])
    bott2[x1] = max(max(y1,y2),bott2[x1])               

img_1 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

g_1 = 0
g_2 = 0
cnt = 0
cnt_2 = 0

for a in range(min_x , max_x):
    min_y = min(up[a],bott[a])
    max_y = max(up[a],bott[a])
    for b in range (min_y , max_y):
        g_1 += img_1[b,a][2]
        cnt += 1

for a in range(min_x_2 , max_x_2):
    min_y = min(up2[a],bott2[a])
    max_y = max(up2[a],bott2[a])
    for b in range (min_y,max_y):
        g_2 += img_2[b,a][1]
        cnt_2 += 1

mean_g1 = (g_1)/cnt
mean_g2 = (g_2)/cnt_2

img_YCRCB= cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
img2_YCRCB= cv2.cvtColor(img2,cv2.COLOR_BGR2YCrCb)

# YCrCb 이미지 채널 실패
#for a in range(len(img)):
#    for b in range(len(img[a])):
#        if img_YCRCB[a,b][1] >=133 and img_YCRCB[a,b][1]<=173and img_YCRCB[a,b][2] >= 77 and img_YCRCB[b,a][2] <=127:
#             img_1[b,a][1] += mean_g1 - mean_g2

#for a in range(len(img2)):
#    for b in range(len(img2[a])):
#        if img2_YCRCB[a,b][1] >=133 and img2_YCRCB[a,b][1]<=173and img2_YCRCB[a,b][2] >= 77 and img2_YCRCB[b,a][2] <=127:
#             img_2[b,a][1] += mean_g2 - mean_g1

#image_3 = cv2.cvtColor(img_1, cv2.COLOR_HSV2BGR)
#image_4 = cv2.cvtColor(img_2, cv2.COLOR_HSV2BGR)
#cv2.imwrite("testY1.png",image_3)
#cv2.imwrite("testY2.png",image_4)

for a in range(min_x , max_x):
    min_y = min(up[a],bott[a])
    max_y = max(up[a],bott[a])
    for b in range(min_y,max_y):
        img_1[b,a][1] += mean_g2 - mean_g1

for a in range(min_x_2 , max_x_2):
    min_y = min(up2[a],bott2[a])
    max_y = max(up2[a],bott2[a])
    for b in range(min_y,max_y):
        img_2[b,a][1] += mean_g2 - mean_g1
       
image_1 = cv2.cvtColor(img_1, cv2.COLOR_HSV2BGR)
image_2 = cv2.cvtColor(img_2, cv2.COLOR_HSV2BGR)


cv2.imwrite("test1.png",image_1)
cv2.imwrite("test2.png",image_2)
cv2.destroyAllWindows()