#IMPORTS
from PrepSteps.ReSize import FindSize ,SizeScaler
from PrepSteps.OneShotAugment import FindMax , AugmentData


def PrepareData(data_path, threshold, left_overs = True):
    
    ''' Gives information about the dataset 
        - Size of smallest image and largest
        - MEAN width and height
        - MEDIAN width and height
        -The most popular image sizes in our dataset
    
    This could help to choose the most appropriate final size of the images
    
    
    
    IMPORTANT :
        The directory should have folders and sub-folders !
        The images should be .PNG (Otherwise change to the appropriate extension)
    '''
    
    FindSize(data_path)
    ####################################################################
    
    '''
    Checks the images based on the given thrushold (tuple)
    Resizes the images based on a final indicated size (tuple)    '''
    print("============================================================\n")
    SizeScaler(data_path)
    
    #################################################################### 
    
    ''' Calculates the maximum possible size of the dataset after 
    the 6 augmentations  
    + removes the folders with smaller size than the indicated one
    
    * Creates a separate folder (set_aside) where the eliminated/smaller folders are located  '''
    print("============================================================\n")
    
    ### Create the path for the left over folders 
    rem = len(data_path) - len(data_path.split('\\')[-1]) - 1
    new_path = data_path[:rem]
    
    if left_overs == True:
        FindMax(new_path, threshold)
    
    ####################################################################
    
    
    #### Augment the data to increase the size
    print("============================================================\n")
    AugmentData(new_path)

