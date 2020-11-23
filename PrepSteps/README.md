--- Not all the steps are mandatory if the dataset is completelly prepared ---

Explanation of the code:
  - Resize.py
    -- FindSize   : Provides some information regarding the images in the dataset (width,lenght,depth, etc)
	-- SizeScaler : Rescales all the images into appropriate size (default :120x120)
    
    
  - OneShotAugment.py
    -- Calculates the threshold for maximizing the given dataset with 6 augmentation steps -> augments and saves the data
    
    
    
  - PrepareData.py
    --
    - If you need all of the preprocessing steps, then call the PrepareData.py that runs all the above in a sequence
    
    
---NOTE---

The scripts work for every dataset if :

  - The images are in .png format (otherwise change to your format of preferance)
  - The folders should have a main folder for each category and subfolders for each one of them
  
      i.e. 
      
          main_dir/Drugs/Opium
          main_dir/Drugs/Cannabis
          main_dir/**/*
  -
