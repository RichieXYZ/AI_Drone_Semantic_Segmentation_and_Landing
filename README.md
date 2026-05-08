# AI_Drone_Semantic_Segmentation_and_Landing
U-Net based semantic segmentation softwares for drone aerial view

Dataset: https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset

Structure:

AI_Drone_Semantic_Segmentation_and_Landing/  
|  
|- Data/  
|- Models/  
|- Results/  
|- DroneDataset.py  
|- ImageAugmentation.py  
|- Landing_point_prediction.py  
|- Testing.py  
|- Training.py  
|- Utility.py  

1. Data/ -> organized in three folders containing the images, integer masks and rgb masks, the size of the download is ~ 4.23 GB, and the class dictionary.



<img width="769" height="295" alt="Screenshot 2026-05-08 at 20 28 44" src="https://github.com/user-attachments/assets/bda4c29a-dc67-46cb-b326-dc8e29964290" />
 
2. Models/ -> the folder where the trained models are stored

3. Results/ -> contain the results of exploratory run to select the best encoder backbone for the model. Different networks have been tested and efficientnetb0 performed the best.

4. DroneDataset.py ->

5. ImageAugmentation.py ->

6. Landing_point_prediction.py
