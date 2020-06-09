# DarkShotLearning + DarkTNN
**One Shot Learning with Siamese Neural Networks on Dark Web Images + Triplet Siamese Neural Networks with Triplet Loss**

The datasets used for the experiements is consisted by 3500 images scraped from the Dark Web
  - There are 11 main categories
  
  
  
An example of the data can be found in the data folder

   The data folder contains 3 datasets that were generated from the same initial images (3500)
    
     - Dummy_Druglord    (25 categories) : By shape        -->  All powders together, all pills together, etc
     
     - Mediocre_Druglord (67 categories) : By_type         -->  Benzos/Attivan, Benzos/Valium
     
     - Advanced_Druglord (100 categories): By_type & shape -->  Benzos/Xanax Pills, Benzos/Xanax Powder, 
                                                                Benzos/Xanax Boxes, etc
     
     
    Note:
     From top to bottom the difficulty/hours of labelling the datasets increased
     Most of the changes are in the drug categories
     Each dataset has different results for the models
     
You can download the full dataset from:
      
      
      
**Code Explanation:**

- **PrepSteps:**
  - **ReSize.py**  --
  - **OneShotAugment.py**  --

- **LoadModels:**
  - **CategoricalNets.py** -- 2 categorical convolutional neural networks (CNN) 
  - **SiameseNet.py**  -- The Siamese Neural Network with Absolute Distance calculation and Binary Cross-entropy Loss
  - **SiameseNet_Contrastive.py** -- The Siamese Neural Network using the Eucledian Distance and Contrastive Loss
  
- **DataLoad:**
  - **PairGen.py** -- Creates balanced N-shot pairs for the Siamese Neural Networks
  - **DataPrep.py** -- Creates random N-shot pairs for the Siamese Neural Networks - The positive and negative representation is                          completely unbalanced
