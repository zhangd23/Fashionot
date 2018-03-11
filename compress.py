import matplotlib.pyplot as plt
from skimage import data, color, io, img_as_ubyte
from skimage.transform import rescale, resize, downscale_local_mean
import os

#open images in ./images

root = '/home/kivy/code/Fashionot/'
subdir ='images/'
path = root+subdir

#get list of files
filelist = os.listdir(path)

for filename in filelist:
	
	astronaut = io.imread(path+filename)
	#change them to 100 x 100 x 16 bits
	#save them in ./images_transformed
	image = img_as_ubyte(astronaut) #convert to 8bit
	image_resized = resize(image, (100,100))#resize to 100x100 pixels
	savepath = '/home/kivy/code/Fashionot/images_transformed/'
	io.imsave(savepath+filename,image_resized)

print 'done'


