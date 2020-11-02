# DarkShotLearning + DarkTNN
**One Shot Learning with Siamese Neural Networks on Dark Web Images
-Triplet Siamese Neural Networks with Triplet Loss**


**Overview**
In this project the Label-Agnostic Siamese and Triplet Siamese Neural Networks are used for One-Shot and Few-Shot experiments. In general, the Siamese Neural Networks are (two) identical neural networks that are trained in parallel. The neural networks produce embeddings (128,256,..,etc) from the input images which are then used to identify the similarities of the two outputs. The similarity of the two embeddings is calculated through the contrastive loss function.

The models were tested on images of various sizes that (mainly) depict illegal drugs. 

The datasets used for the experiments are consisted by 3500 images scraped from the Dark Web
  - There are 11 main categories
  
  
  
An example of the data can be found in the data folder

   The data folder contains 3 datasets that were generated from the same initial images (3500)
    
     - Junior_Dealer (Sample Included)   (25 categories) : By shape        -->  All powders together, all pills together, etc
     
     - Senior_Dealer (Not Included)      (67 categories) : By_type         -->  Benzos/Attivan, Benzos/Valium
     
     - Druglord (Sample Included)        (100 categories): By_type & shape -->  Benzos/Xanax Pills, Benzos/Xanax Powder, 
                                                                Benzos/Xanax Boxes, etc
     
     
    Note:
     From top to bottom the difficulty/hours of labelling the datasets increased
     Most of the changes are in the drug categories
     Each dataset has different results for the models
     
      
      
      
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
  - **TripletGen.py** -- Creates balanced N-shot triplets fro the Triplet Siamese Network
  
- **TripletSiamese:**
  - **Triplet_Loss.py** -- Calculates the Triplet Loss for the Siamese Neural Network
  - **TripletNet.py** -- The Triplet Siamese Neural Network
  
__________________________________________________________________________________________________________________________________
- **TO RUN THE MODELS**
  - **Test_Triplet.ipynb**  -- Jupyter Notebook with the Training and Testing of the Tiplet Siamese Neural Network
  - **Test_Siamese.ipynb**  -- Jupyter Notebook with the Training and Testing of the Siamese Neural Network
  - **Test_Baseline.ipynb** -- Jupyter Notebook with the baseline comparison models (CNN + KNN + NAIVE-statistical)
  
  
  
- **Figures**
  - Includes the TSNE manifolds from the 10-way and 8-way datasets
  
    
    ![alt text](https://github.com/DimosMellios/DarkShotLearning/blob/master/Figures/8-way-100epochs.png)

