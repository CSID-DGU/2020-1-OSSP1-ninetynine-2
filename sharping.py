import cv2

input_imgs =cv2.imread("sample6.jpg",cv2.IMREAD_COLOR)
input_HSV = cv2.cvtcolor(input_imgs,cv2.COLOR_BGR2HSV)

height, width, channel = image.shape

v = 0

for i in range(height):
    for j in range(width):
        v+=input_HSV[j][i][2]
v = int( v / (height*width))

b, g, r = cv2.split(input_imgs)

blur_b=cv2.GaussianBlur(b,(5,5),3)
blur_g=cv2.GaussianBlur(g,(5,5),3)
blur_r=cv2.GaussianBlur(r,(5,5),3)

sub_b=cv2.subtract(b,blur_b)
sub_g=cv2.subtract(g,blur_g)
sub_r=cv2.subtract(r,blur_r)

add_b=cv2.add(b,sub_b)
add_g=cv2.add(g,sub_g)
add_r=cv2.add(r,sub_r)

b=cv2.equalizeHist(add_b)
g=cv2.equalizeHist(add_g)
r=cv2.equalizeHist(add_r)

if(v < 100):
    b=cv2.add(b,10)
    g=cv2.add(g,10)
    r=cv2.add(r,10)

result_imgs=cv2.merge((b,g,r))
result_imgs=cv2.fastNlMeansDenoisingColored(result_imgs)
cv2.imwrite("result6.jpg",result_imgs)

cv2.waitKey(0)
cv2.destroyAllWindows()
