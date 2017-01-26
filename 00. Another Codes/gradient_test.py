from PIL import Image
from numpy import *
from pylab import *

import warp

im1 = array(Image.open('D:\\ppy\\beatles.png').convert('L'))
im2 = array(Image.open('D:\\ppy\\billboard_for_rent.jpg').convert('L'))

tp = array([[264,538,540,264],[40,36,605,605],[1,1,1,1]]) # 벽보의 좌표

im3 = warp.image_in_image(im1,im2,tp)

figure(1)
gray()
imshow(im1)
axis('equal')
axis('off')
show()

figure(1)
gray()
imshow(im2)
axis('equal')
axis('off')
show()

figure(1)
gray()
imshow(im3)
axis('equal')
axis('off')
show()
