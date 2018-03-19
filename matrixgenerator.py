#this script generates a matrix from our training data
#X matrix is our training data, Y matrix is our labels

import os
from skimage import io, img_as_ubyte
import numpy as np

    
def generateMatrices():
    X = np.empty((0,30000), 'uint8')
    Y = np.empty((0,1),'uint8')	
    #for all images in folder 1    
    
        # Root directory 
    path = os.path.dirname(os.path.realpath('matrixgenerator.py'))
    
    # Folder paths
    folder1 = path+r'\images_transformed\fashionable' #positive label ==1
    folder2 = path+r'\images_transformed\unfashionable'#negative label ==0
    
    # Create Folders
    if not os.path.exists(folder1):
        os.makedirs(folder1)
    
    if not os.path.exists(folder2):
        os.makedirs(folder2)
        
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

generateMatrices()
