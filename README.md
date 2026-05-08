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

<img width="1072" height="432" alt="Screenshot 2026-05-08 at 20 53 39" src="https://github.com/user-attachments/assets/e85f9e6c-6cc5-4464-b8c5-57023e4cc66c" />

4. DroneDataset.py ->

5. ImageAugmentation.py ->

6. Landing_point_prediction.py -> this script defines the algorithm for the computation of the best landing point. It works on the segmentation map by selecting the center of the largest circle inscribed in the paved area or grass. 

7. Testing.py -> the script for running inference of the model on the test set. Here are monitored the per-class accuracy, average accuracy and average IoU.

e<img width="1180" height="378" alt="Screenshot 2026-05-08 at 22 04 55" src="https://github.com/user-attachments/assets/e4a1d233-0808-4581-abfb-86e241c80104" />


<img width="900" height="612" alt="Screenshot 2026-05-08 at 22 02 07" src="https://github.com/user-attachments/assets/b6b30e07-9f46-4d1d-bb78-d513c6c099eb" />


9. Training.py ->

10. Utility.py ->
