
# Remove unused libraries
from skimage import io, img_as_ubyte
from skimage.transform import resize
import os
from PIL import Image

# Root directory 
path = os.path.dirname(os.path.realpath('compress.py'))

# Folder paths
folder1 = path+r'\images_transformed\fashionable'
folder2 = path+r'\images_transformed\unfashionable'

# Create Folders
if not os.path.exists(folder1):
    os.makedirs(folder1)

if not os.path.exists(folder2):
    os.makedirs(folder2)
    
#open images in ./images    
imgpath1 = path+r'\images\fashionable'
imgpath2 = path+r'\images\unfashionable'


#compress function
def compress_image(imgpath, savpath):
    #get list of files
    filelist = os.listdir(imgpath)

    for filename in filelist:
        #os.chmod(imgpath, 0o777)
        astronaut = io.imread(os.path.join(imgpath,filename))
        #change them to 100 x 100 x 16 bits
        #save them in ./images_transformed
        image = img_as_ubyte(astronaut) #convert to 8bit
        image_resized = resize(image, (100,100)) #resize to 100x100 pixels
        savepath = savpath
        io.imsave(os.path.join(savepath,filename),image_resized)

compress_image(imgpath1, folder1)
compress_image(imgpath2, folder2)

# Known Issue: compress_image fails if image is .GIF
# To Do: Check for GIFs and delete

def mirror_image(imgpath, savpath):
    #get list of files
    filelist = os.listdir(imgpath)

    for filename in filelist:
        image_obj = Image.open(os.path.join(imgpath,filename))
        mirror_image = image_obj.transpose(Image.FLIP_LEFT_RIGHT)
        savepath = savpath        
        new_filename = filename[:-4] + 'm' + filename[6:]            
        mirror_image.save(os.path.join(savepath,new_filename))

mirror_image(folder1, folder1)
mirror_image(folder2, folder2)

print ('Done')

