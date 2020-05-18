#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# IMPORTS
import glob
import pandas as pd
import numpy as np
import statistics
import os , random
import shutil
import errno
import imageio
import imgaug as ia
import numpy as np
import cv2
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt

from tqdm.auto import tqdm


# In[43]:


def FindMax(location, at_least = None):
    
    '''
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    + Provides calculation based on the size of images the user indicates        +
    + At least X images in each of the folders                                   +
    +                                                                            +
    + Prints the number of categories left based on the threshold the user gives +
    +                                                                            +
    +  IMPORTANT:                                                                +
    +                                                                            +
    +  if the user specifies a number then the split is made as well             +
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    '''
    location = location + '\\Train_Data'
    
    ## Find the images in the folder
    t = glob.glob(location + '\***\**\*.png')

    ## Save the locations
    df = pd.DataFrame(t)
    df.columns = ['Path']
    df['Category'] = df['Sub-Category'] = ''

    ## Split in categories and sub-categories
    for i in range(0,len(df)):
        #--
        tmp_categories = df['Path'][i].split(location)[1].split("\\")[1:3]
        df['Category'].iloc[i] = tmp_categories[0]
        df['Sub-Category'].iloc[i] = tmp_categories[1]
    
    df['Sub-Category'] = df['Category'] + '\\' + df['Sub-Category']
    
    tmp_list = list(df['Sub-Category'])
    
    ### Count the size of each category in our dataset
    my_dict = {i:tmp_list.count(i) for i in tmp_list}
    
    ### Sort the counted categories
    new_dict = {k: v for k, v in sorted(my_dict.items(), key=lambda item: item[1],reverse=True)}
    
    ### Store the keys and values separatelly
    v = list(new_dict.values())
    k = list(new_dict.keys())
    
    # Store in dataset
    counter = pd.DataFrame(k,columns = ['Names'] )
    counter['Counts'] = v
    
    
    
    '''
       ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
       + Provides the best possible combination of remaining categories + 
       + for the size maximization of the final dataset                 +
       ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
       
    '''            
    
    ### Find where the dataset maximizes after the 6 augmentation multiplications
    size = 0
    tmp = []
    
    for i in range(5,max(v)):
        measure = len(counter[counter['Counts'] >=i])
        size_new = (i*6+i)*measure

        if size_new > size:
        
            size = size_new
            categories_left = measure
            k = i 
            tmp.append(size_new)
            
        else:
            pass
        
    print('The maximum size this dataset could be, after the augmentations is ',size,
          '\nIf we keep the categories with at least ({}) images'.format(k), 
          '\n\nThe final number of categories left will be ' ,categories_left)
    
    
    #######################################################################
    ## Keeps the dataset as specified from the user
    
    if at_least is not None:
        
        set_aside = counter[counter['Counts'] < at_least]
        set_aside = set_aside.reset_index(drop=True)
        
        keep_them = counter[counter['Counts'] >= at_least]
        keep_them = keep_them.reset_index(drop=True)
        
        print('\n==============================================')
        print('If you need at least {} images in each folder:'.format(at_least))
        print('\nThe number of categories left are :', len(keep_them), 
              '\n\nAnd the set aside categories are :', len( set_aside))
        
        try:
            
            removed = location.split('\\')[-1]
            new_directory = location.replace(removed,'') + 'Test_Data'
            os.mkdir(new_directory)
            
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            pass
        
        ## Split the names of the initial directory to make the new folder names
        old_folders = location + '\\' + set_aside['Names']
        
        
              
        ################## Creating the Folders #####################

        tmp_old_folders = []
        
        for j in old_folders:
            folders = j.split('\\')[-2]
            tmp_old_folders.append(folders)
            
        
        ## Create the folders in the new destination   
        for t in tmp_old_folders:

            try:
                os.mkdir(new_directory + '\\' + t)
        
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
                pass
        
        
        ####### Make the new sub-folders 
        new_loc = new_directory +'\\'+ set_aside['Names']

        for dest in new_loc:

            try:
                os.mkdir(dest)
        
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
                pass
        
        
         ####### Find the files that have to be moved ##########
        tmp_files = []
        for f in old_folders:
            file = glob.glob(os.path.join(f,'*.png'))
            tmp_files.append(file)

            
        ######## We must flatten the list of lists ##########
        flat_list = [item for sublist in tmp_files for item in sublist]
        
        new_folders = []
        for i in range(0,len(flat_list)):
            
            CAT, SUBCAT = flat_list[i].split(location)[1].split('\\')[1:3]
            tmp_new_folders = os.path.join(new_directory,CAT,SUBCAT)
            new_folders.append(tmp_new_folders)
            
          
        ######## move the files to the new folder ###########
        print('====================================================')
        print('\nMoving the unwanted images to a separate folder...')  
        
        for m in range(0,len(flat_list)):
            shutil.move(flat_list[m],new_folders[m])
        
        print('DONE')
        
        ####### remove the images from the old directory ###########
        print('====================================================')
        print('\nRemoving the empty folders...')
        
        old_directory = location + '\\' + set_aside['Names']
        
        for r in old_directory:
            shutil.rmtree(r)
            
        print('DONE')
        
        print('\n\nThere are {} moved files and the new directory is :\n\n'.format(len(flat_list)), new_directory)

    else:
        print('Note: You did not specify a threshold for the minimum acceptable number of images per category')


# In[44]:


def counter(data,location):
    
    '''Creates a dataframe with the location and
    the category + sub-category information'''
    
    df = pd.DataFrame(data)
    df.columns = ['Path']
    df['Category'] = df['Sub-Category'] = ''

    ## Split in categories and sub-categories
    for i in tqdm(range(0,len(df))):

        tmp_categories = df['Path'][i].split(location)[1].split("\\")[1:3]
        df['Category'].iat[i] = tmp_categories[0]
        df['Sub-Category'].iat[i] = tmp_categories[1]
    
    df['Sub-Category'] = df['Category'] + '\\' + df['Sub-Category']
    
    return df


# In[45]:


def AugmentSteps(elem,l,r):
    
    '''Augments the images in 6 ways
    Inputs:
        - image location
        - directory location
        - exact file location
        '''
    
    
    # read the image
    image = cv2.imread(elem)
    # image = imageio.imread(elem)
            
    # 1st augmentation --> ROTATE
            
    rotate=iaa.Affine(rotate=(-50, 30))
    rotated_image=rotate.augment_image(image)
    
    cv2.imwrite('{}\\rotated{}.png'.format(l,r), rotated_image) 
            
    # 2nd augmentation --> FLIP HORIZONTALLY
            
    flip_hr=iaa.Fliplr(p=1.0)
    flip_hr_image= flip_hr.augment_image(image)
    cv2.imwrite('{}\\flipped_hor{}.png'.format(l,r), flip_hr_image)
    #plt.imsave('{}\\flipped_hor{}.png'.format(l,r), flip_hr_image, cmap='Greys')
            
    # 3rd augmentation --> FLIP VERTICALLY
            
    flip_vr=iaa.Flipud(p=1.0)
    flip_vr_image= flip_vr.augment_image(image)
    cv2.imwrite('{}\\flipped_ver{}.png'.format(l,r), flip_vr_image)
            
    # 4th augmentation --> CROP
            
    crop = iaa.Crop(percent=(0, 0.3)) # crop image
    crop_image=crop.augment_image(image)
    cv2.imwrite('{}\\cropped{}.png'.format(l,r), crop_image)
            
    # 5th augmentation --> CHANGE CONTRAST
            
    contrast=iaa.GammaContrast(gamma=2.0)
    contrast_image =contrast.augment_image(image)
    cv2.imwrite('{}\\contrast_change{}.png'.format(l,r), contrast_image)
            
    # 6th augmentation --> ADD NOISE
    gaussian_noise=iaa.AdditiveGaussianNoise(10,25)
    noise_image=gaussian_noise.augment_image(image)
    cv2.imwrite('{}\\noise_add{}.png'.format(l,r), noise_image )


# In[46]:


def AugmentData(location):
    ''' 
     +++++++++++++++++++++++++++++++++++++++++++++++++++++
     +  Collects the directories of the images           +
     +                                                   +
     +  + Augments the images with the appropriate name  +
     +  + Cleans the dataset                             +
     +    -- Randomly pick overpopulated images          +
     +    -- Deletes those images                        +
     +++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    '''
    
    ### Directory
    location = location + '\\Train_Data'
    
    # Find the initial images
    initial_images = glob.glob(location + '\\***\**\*.png')
    
    ## Save the locations with the folder info 
    df = counter(initial_images,location)
    
    sub_list = df.groupby('Sub-Category').count().reset_index().sort_values(by=['Path']).reset_index(drop=True)
    
    ##### prepare the directories for glob.glob
    locations_list = [location +'\\'+ x  for x in sub_list['Sub-Category']]
    extension_list = [x + '\\*.png' for x in locations_list]
    
    ################ Start Augmentation from the smallest folder ###################
    print('\nThe Augmentation Started.')
    print('\nThis will take a minute...')
    
    for j in tqdm(range(0,len(extension_list))):
        tmp = glob.glob(extension_list[j])
        
        ##### AUGMENTATION STEPS #####
        r = 0
        for elem in tmp:
            l = locations_list[j]
            
            AugmentSteps(elem,l,r)
            
            r += 1
    
    print('\nCleaning the dataset...')
    
    
    augmented_images = glob.glob(location + '\\***\**\*.png')  
    df2 = counter(augmented_images, location)
    
    ### Count the size of each category folder
    grouped = df2.groupby('Sub-Category').count().reset_index().sort_values(by=['Path']).reset_index(drop=True)
    
    #### The smallest folder is the indicator for the size of the folders
    grouped['Category'] = grouped['Category'] - grouped['Category'][0]
    
    #### Remove it and we have the number of the images that we have to remove 
    grouped['Sub-Category'] = location + "\\" +grouped['Sub-Category']
    
    #### Randomly pick an image an remove it
    print('\nPreparing some final changes. . .')
    
    for i in range(1,len(grouped)):
        size = grouped['Category'][i]
        
        for j in range(0,size):
            tmp_f = grouped['Sub-Category'].iloc[i]
            # Picks random image from the folder
            ran_choice = random.choice(os.listdir(tmp_f))
            # Removes the image 
            os.unlink(os.path.join(tmp_f,ran_choice))
    
    #### Give an example with the augmentations ######
    
    
    print('ALL DONE')

