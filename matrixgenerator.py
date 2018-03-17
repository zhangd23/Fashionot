#this script generates a matrix from our training data
#X matrix is our training data, Y matrix is our labels

import os
from skimage import io, img_as_ubyte
import numpy as np

    
def generateMatrices():
    X = np.empty((0,30000), 'uint8')
    Y = np.empty((0),'uint8')	
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
        
    tmplabel = 1 #folder 1 is positive examples
    for imagefilename in os.listdir(folder1):
      #load image
      image = io.imread(os.path.join(folder1,imagefilename))
      image = img_as_ubyte(image) #convert to 8bit
      #reshape image newshape
      datarow = np.reshape(image, (1,-1)) # make a 1 x m matrix where m is however many pixels
                                # there are in the image
      X=np.append(X,datarow,axis=0)
      Y=np.append(Y,np.array([tmplabel]),axis=0)
    
    
    
    #repeat for folder2
    tmplabel = 0 #folder 2 is negative examples
    for imagefilename in os.listdir(folder2):
      #load image
      image = io.imread(os.path.join(folder2,imagefilename))
      image = img_as_ubyte(image) #convert to 8bit
      #reshape image newshape
      datarow = np.reshape(image, (1,-1)) # make a 1 x m matrix where m is however many pixels
                                # there are in the image
      X=np.append(X,datarow,axis=0) #append onto the same matrix
      Y=np.append(Y,np.array([tmplabel]),axis=0)
      
    print(np.shape(X))
    print(np.shape(Y))
    return X,Y

generateMatrices()
