
import matplotlib.pyplot as plt
from skimage import data, color, io, img_as_ubyte
from skimage.transform import rescale, resize, downscale_local_mean
import os

#open images in ./images

root = '/home/kivy/code/Fashionot/'
subdir ='images/'
filename = root + subdir + '000001.jpg'
astronaut = io.imread(filename)


#image = color.rgb2gray(astronaut)
image = img_as_ubyte(astronaut) #convert to 8bit

image_rescaled = rescale(image, 1.0 / 4.0)
image_resized = resize(image, (100,100))

fig, axes = plt.subplots(nrows=1, ncols=2)

ax = axes.ravel()

ax[0].imshow(image)
ax[0].set_title("Original image")

ax[1].imshow(image_resized)
ax[1].set_title("Resized image (no aliasing)")

ax[0].set_xlim(0, 512)
ax[0].set_ylim(512, 0)
plt.tight_layout()
plt.show()


#change them to 100 x 100 x 16 bits
#save them in ./images_transformed
