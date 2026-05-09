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

5. ImageAugmentation.py -> This scripts provides a separate sandbox to design and try image augmentation using the package Albumentations for this project. In fact, in this setup, augmentation is not only useful to enlarge the data available for training, but provides a whole system to enhance the robustness of the drone capability to handle unseen situations. The provided augmentations, beside ordinary geometrical transformation such as resizing, flipping and affine transforms, account for different weather conditions and eventual camera failures like dead pixels or sensor damage.

<img width="1168" height="416" alt="Screenshot 2026-05-09 at 12 57 42" src="https://github.com/user-attachments/assets/ec93ac2d-1398-4dc9-be58-e1cd33a613c1" />

6. Landing_point_prediction.py -> this script defines the algorithm for the computation of the best landing point. It works on the segmentation map by selecting the center of the largest circle inscribed in the paved area or grass. 

7. Testing.py -> the script for running inference of the model on the test set. Here are monitored the per-class accuracy, average accuracy and average IoU.  

<img width="1180" height="378" alt="Screenshot 2026-05-08 at 22 04 55" src="https://github.com/user-attachments/assets/e4a1d233-0808-4581-abfb-86e241c80104" />

<img width="900" height="612" alt="Screenshot 2026-05-08 at 22 02 07" src="https://github.com/user-attachments/assets/b6b30e07-9f46-4d1d-bb78-d513c6c099eb" />  

8. Training.py -> in this script are defined the functions for training the model. In particular the function encoder_comparison is used as a preliminary training of 30 epochs in order to select the best encoder backbone among few architectures : ['mobilenet_v2', 'resnet50', 'efficientnet-b0', 'densenet121']. Once the best encoder is selected, it is employed in a U-Net++ architecture for the actual training with early stopping.  

<img width="859" height="496" alt="Screenshot 2026-05-09 at 12 41 33" src="https://github.com/user-attachments/assets/92034123-7513-4ce5-9378-3a6c152d59ca" />  

The U-Net++ is a convolutional encoder-decoder architecture based on the ordinary U-Net but with the addition of nested skip connections that aggregate the features while decoding the encoded data, and deep supervision that inject gradients at all depths providing a regularization in the deeper layers of the model. This architecture is specifically built for semantic segmentation tasks.

9. Utility.py -> This file contains all the utility functions uded in the scripts.
