# Cats and Dogs Image Classification
[Report](https://github.com/Aryan-IIT/data_augmentation_cs203/blob/main/Documentation_Lab05_Assignment.pdf)  
## Task 1: Data Preparation  

### Step 1: Download the Dataset  
Download the dataset from Kaggle:  
[Dataset Link](https://www.kaggle.com/datasets/samuelcortinhas/cats-and-dogs-image-classification?select=test)  

**Important:** Download **only the `test` set**.

### Step 2: Split the Dataset  
Run `dataset_builder.py` to split the dataset into **80% training** and **20% testing**.

### Step 3: Augment the Training Set  
Run `augmentor.py` to apply various data augmentation techniques to the training set.

### Step 4: Verify Data Processing  
Use `verifying.ipynb` to generate plots and visualizations to confirm dataset correctness.

This concludes Task 1.

---

## Task 2: Model Training & Evaluation  

### Step 5: Define the Model  
The model architecture is defined in `model.py`, which creates a **ResNet50** classifier.

### Step 6: Train and Evaluate  
Run `training_and_inference.ipynb` to:  
- Train **two models** (one on the original dataset, one on the augmented dataset).  
- Compute performance metrics: **accuracy, precision, recall, and F1-score**.  
- Compare results and analyze model effectiveness.

### Documentation 
- Interpreataion and documentation is mentioned in `Documentation_Lab05_Assignment.pdf`
