# IMPORTS
import os
import errno
import glob
from tqdm.auto import tqdm
import pandas as pd
from PIL import Image
import shutil
import statistics


class color():
    '''Just to make some part of the printed parts bold'''
    BOLD = '\033[1m'
    END = '\033[0m'



def get_key(val,my_dict): 
    tmp_dict = my_dict
    for key, value in tmp_dict.items(): 
        if val == value: 
            return key 

def FindSize(location):
    
    '''It gives some information about the dataset 
    Size of images, the most popular sizes in your dataset etc'''
    
    # Find the locations of the images in the subfolders and save paths in a dataframe
    t = glob.glob('{}\***\**\*.png'.format(location))
    
    print('Our dataset is consisted by ',len(t),' images')
    
    
    # find the sizes of the various images
    size_list = []
    
    for image_path in t:
        image = Image.open(image_path)
        size_list.append(image.size)
    print('\nThe biggest image in our list is :', max(size_list),
          ' and the smallest is:', min(size_list))
    
    
    # Separate the width and height of the images
    width_list, height_list = zip(*size_list)
    
    print(color.BOLD,'\nSummary:',color.END)
    print('============================')
    print('Mean width: ', statistics.mean(width_list))
    print('Mean height: ', statistics.mean(height_list))
    print('Median Width: ', statistics.median(width_list))
    print('Median Height: ', statistics.median(height_list))
    print('============================')
    
    my_dict = {i:size_list.count(i) for i in size_list}
    
    # Collect only the count of the sizes in order to find the 5 most popular
    the_values = my_dict.values()
    Values = list(the_values)
    # Sort and keep the 5 first
    Values.sort(reverse=True)
    Values = Values[:5]
    
    j = 1 
    print(color.BOLD, '\nTop 5 Sizes:' ,color.END)
    print('=====================================')
    for i in Values:
        print('No', j,'-->', get_key(i,my_dict), 'with  ', i, 'appearances')
        j +=1
        

def SizeScaler(source, thresh_size = None,final_size = None):
    
    '''Checks if the images meet the given threshold
       - Resizes keeping the aspect ratio
       
       - Resizes to squared images based on the final size 
       - If no size is specified the images are resized to 120x120 pixels'''
    
    
    ################# Destination Folder ############################
    
    one_folder_back = source.split('\\')[-1] 
    destination = source[:-(len(one_folder_back)+1)]
    
    
    # Locate the images and store in dataframe with folders
    image_paths = glob.glob('{}\***\**\*.png'.format(source))
    df = pd.DataFrame(image_paths)
    df.columns = ['Path']
#     df['Category'] = df['Sub-Category'] = ''
    
    tmp_category = []
    tmp_sub_cat  = []
    final_loc  = []
    
    # split to the category and sub-category
    for im in df['Path'] :
        
        tmp_name = im.split(source)[1]
        #The name of the image
        remover = tmp_name.split('\\')[3]
        
        sub_cat = tmp_name.rsplit(remover, 1)[0]
        categ = sub_cat.split('\\')[1]
        
        tmp_category.append(categ)
        tmp_sub_cat.append(sub_cat)
        
        # List with the final destination of each image
        
        
        new_location = destination + '\\Train_Data' + tmp_name
        final_loc.append(new_location)
    
    
    print('Creating the folders and sub folders')
    # Create the folders for the images
    try:
        os.mkdir(destination + '\\Train_Data')
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    
    # Create folders for categories
    for j in tmp_category:
        try:
            os.mkdir(destination + '\\Train_Data\\' + j)
            
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            pass
        
    # Create sub-folders
    for s in tmp_sub_cat:
        try:
            os.mkdir(destination + '\\Train_Data\\' + s)
            
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            pass
        
    print('Done !\n') 
    
    print('Please wait the images are being moved to the new location...\n')
    
    print('This will take a moment or two...')
    print('_________________________________')
    
    # Copy the images in the new folders
    for i in tqdm(range(0,len(image_paths))):
    
        initial_loc = image_paths[i]
        destin_loc  = final_loc[i]
        # Copy gto the new location
        dest = shutil.copy(initial_loc, destin_loc)  
        
    print('\n{} images moved successfuly to the new folders'.format(i+1))
    
    print('\n\nResizing the images in the destination folder...')
    
    new_image_paths = glob.glob('{}\***\**\*.png'.format(destination+'\\Train_Data'))
     
    
    ###########  Scaling the smaller images to an appropriate size  ################
    if thresh_size == None:
        thresh_size = (120,120)
        print('\nThe threshold for the minimum size of image is 120x120\n')
        
        for image_path in new_image_paths:
            
            image = Image.open(image_path)
            
            # split the dimmetions of the images
            w,h = zip((image.size))
            
            if w[0] >= 120 and h[0] >= 120:
                pass
            
            else:
            
                new_image = image.resize((w[0]*2,h[0]*2))
                new_image.save(image_path, 'PNG')
                
    else:
        print('\nYou have specified the minimum size of acceptable images to {}\n'.format(thresh_size))
        for image_path in new_image_paths:
            # split the dimmetions of the images
            w,h = zip((thresh_size))
            
            if w[0] >= 120 and h[0] >= 120:
                pass
            
            else:
                print('no')
                new_image = image.resize((w[0]*2,h[0]*2))
                new_image.save(image_path, 'PNG')
                
            
    print('All the images were tested to fit the specified threshold --> ', thresh_size)
    
    ### Find the changed images once again
    new_image_paths = glob.glob('{}\***\**\*.png'.format(destination+'\\Train_Data'))
    
    print('\n======================================================================\n')
    print('The images are changing to the apropriate size ', final_size)
    
    ###########  Change all the images to squared sized images  ################
    
    # -------- We can change the size to rectangulars if we specify a size ------- #
    
    if final_size == None:
        
        for image_path in new_image_paths:
            
            image = Image.open(image_path)
            new_image = image.resize((120,120))
            new_image.save(image_path, 'PNG')
            
        print('\nSince no final size was specified, the images were changed to the standard size --> 120x120\n')
    
    else:
        
        for image_path in new_image_paths:
            
            w2,h2 = zip((final_size))
            
            image = Image.open(image_path)
            new_image = image.resize((w2[0],h2[0]))
            new_image.save(image_path, 'PNG')
            
        print('\nThe images were changed to the specified final size SUCCESSFULLY--> ', final_size)
        print(color.BOLD,'The Dataset is ready to be fed to the model',color.END)
