
# Remove unused libraries
from skimage import io, img_as_ubyte
from skimage.transform import resize
import os
from PIL import Image
import random

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
    
# Open images in ./images    
imgpath1 = path+r'\images\fashionable'
imgpath2 = path+r'\images\unfashionable'


# Compress function
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

#Mirror function
def mirror_image(imgpath, savpath):
    #get list of files
    filelist = os.listdir(imgpath)

    for filename in filelist:
        image_obj = Image.open(os.path.join(imgpath,filename))
        mirror_image = image_obj.transpose(Image.FLIP_LEFT_RIGHT)
        savepath = savpath        
        new_filename = filename[:-4] + 'm' + filename[6:]            
        mirror_image.save(os.path.join(savepath,new_filename))

# Separate data into Training, Cross-Validation, and Test
# Create 3 lists that randomly shuffle filenames of data set
def split_dataset(filepath):
    filelist = os.listdir(filepath)
    total_num = len(filelist)
    train_num = int(round(len(filelist)*.7))
    xval_num = int(round(len(filelist)*.2))
    
    # randomize filelist
    random.shuffle(filelist)    
    
    train_set = filelist[0:train_num]
    xval_set = filelist[train_num: train_num+xval_num]
    test_set = filelist[train_num+xval_num: total_num]
    
    #return 3 lists that contain filenames of data
    return train_set, xval_set, test_set

compress_image(imgpath1, folder1)
compress_image(imgpath2, folder2)

# Known Issue: compress_image fails if image is .GIF
# To Do: Check for GIFs and delete

mirror_image(folder1, folder1)
mirror_image(folder2, folder2)

#Fashionable dataset, y = 1
split_dataset(folder1)
#Un-fashionable dataset, y = 0
split_dataset(folder2)



print ('Done')

