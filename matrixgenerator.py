#this script generates a matrix from our training data
#X matrix is our training data, Y matrix is our labels

import os
from skimage import io, img_as_ubyte
import numpy as np
import random

# Root directory 
path = os.path.dirname(os.path.realpath('matrixgenerator.py'))

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


def save_to_file(filepath, dataset, name):
    with open(os.path.join(filepath,name) + '.txt', mode='wt', encoding='utf-8') as myfile:
        for lines in dataset:
            #print(lines)
            myfile.write(lines)
            myfile.write('\n')

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
    
    save_to_file(filepath, train_set, 'train_set')
    save_to_file(filepath, xval_set, 'xval_set')
    save_to_file(filepath, test_set, 'test_set')
    
    #return 3 lists that contain filenames of data
    return train_set, xval_set, test_set

# To Do: Make generateMatrices() grab from train_set only
    
def generateMatrices():
    X = np.empty((0,30000), 'uint8')
    Y = np.empty((0,1),'uint8')	
    #for all images in folder 1    
          
    tmplabel = np.ones([1,1]) #folder 1 is positive examples
    for imagefilename in os.listdir(folder1):
      #load image
      image = io.imread(os.path.join(folder1,imagefilename))
      #convert the data from 0-255 to -1 to 1
      image = (2/255)*image
      image = image -1
      #reshape image newshape
      datarow = np.reshape(image, (1,-1)) # make a 1 x m matrix where m is however many pixels
                                # there are in the image    
      X=np.append(X,datarow,axis=0)
      Y=np.append(Y,tmplabel,axis=0)
           
    #repeat for folder2
    tmplabel = np.zeros([1,1]) #folder 2 is negative examples
    for imagefilename in os.listdir(folder2):
      #load image
      image = io.imread(os.path.join(folder2,imagefilename))
      image = img_as_ubyte(image) #convert to 8bit
      image = (2/255)*image
      image = image -1
      
      #reshape image newshape
      datarow = np.reshape(image, (1,-1)) # make a 1 x m matrix where m is however many pixels
                                # there are in the image
      X=np.append(X,datarow,axis=0) #append onto the same matrix
      Y=np.append(Y,tmplabel,axis=0)
      
    return X,Y

#Fashionable dataset, y = 1
split_dataset(folder1)
#Un-fashionable dataset, y = 0
split_dataset(folder2)


generateMatrices()
