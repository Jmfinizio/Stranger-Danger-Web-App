## *Johnathan Finizio, Neeza Singh, Ria Sonalker, 2025-April-28 v1.0.1*

## Overview

Anxiety disorders are one of the most common mental health problems in adulthood. Behavioral inhibition, an early personality trait characterized by fear and withdrawl, can be used as a predictor for anxiety later in life. The reactions that a child has to new situations can be studied to assess which children are at a higher risk for anxiety, but behavioral coding of these experiments is a very lengthy process that can stall research for years at a time. The goal of this project is to determine how machine learning can automate behavioral coding and predict childhood risk of anxiety more efficiently.

Our client, Kathy Sem, is a part of the Biobehavioral and Social-Emotional Development Lab at BU. They have over 600 video between BU and Penn that need manual labeling for proiximity, fear, and freezing behaviors. All of these behaviors are labeled on a scale of 0-2 based on intensity. This process is very lengthy, with the labeling of 1 video taking multiple days. Each video requires at least 2 reviewers, taking up a lot of the lab staff's time.

### A. Provide a solution in terms of human actions to confirm if the task is within the scope of automation through AI.

Performing psychological analysis on video data is very time consuming, when having to consider different aspects of the subjects movements and position in close detail. The aim of this project is to automate data collection from video taped psychological experiments in order to streamline the research process.

### B. Problem Statement:

The goal is to determine how computer vision can be used to create a model processes second by second video frames and detects the position of the subject relative to the stranger and their mother, signs of the subject "freezing", and interactions between the subject and the stranger. 

### C. Completed Steps

* Tuned a YOLO detection model to detect the child, parent, and stranger in the experiment video frames
* Created a data pipeline with computer vision models to...
  * Detect the child/parent/stranger
  * Detect the child's pose to calculate its movement (and compare it to the average freeplay movement)
  * Detect the movement in the child's facial expressions
  * Calculate the distance between the child and the parent/stranger
  * Crop the video into freeplay and experiment sections
    * We analyze the freeplay to get a baseline on the child's movement, and use this to calculate a movement ratio for experiment frames
  * Extract and process second by second frames for the input video
* Cleaned all provided client labels to convert them to the desired continuous second by second format
* Created 3 decision tree classifiers so that we can use our extracted features to predict the desired client output (Proximity, fear, freeze)
* Created a web app that combines the pipeline, decision tree classifiers, and user friendliness
  * The user can upload local videos for analysis with the click of a button

### D. Blockers Faced

* Not all data was received at the same time, and labels were received very late in the process.
  * Solution: Priortized feature extraction and research early on, so later work could focus on model tuning and decision tree creation once labels were received. We also created low-level models on the pilot data, that could be easily scaled to fit larger amounts of data once received.

### E. Next Tasks

* Label more video frames using CVAT (draw bounding boxes) to further train the detection model. Use more than just BU videos (Penn and UNC as well)
* Fix some of the UI in the web app include cancel button and removing sharepoint configuration (even though sharepoint doesn't work, the client will be moving on from using sharepoint so it is not needed)
* Add more features to the pipeline for better classification accuracy (mainly for fear) > 90% is ideal.

## Resources
Our repository is split up into 3 folders; ciss-webapp, pipeline, and creating classifiers

* The ciss-webapp folder contains all functionality from the pipeline and classification models, combining them into a user friendly format. This folder also contains all necessary models. 
* The pipeline folder contains the full pipeline code for feature extraction, along with our training for the YOLO detection model.
* The creating classifiers folder contains our code to clean the client labels, extract features from the videos, and to construct the decision tree classifiers. This folder also contains the raw client labeled, cleaned client labels, and feature extraction outputs (decision tree data)

### Code
* CISS Web App
  * The ciss-webapp is the main platform to interact with our pipeline for our user. It combines all pipeline and classifier functionality, ensuring the correct desired output for the user.
  *Note: Running locally is significantly faster than running through hugging face*
  * Link to Hugging Face deployment:
  
    https://huggingface.co/spaces/spark-ds549/CISS-Web-App?logs=container     
  * Link to folder:

    https://github.com/BU-Spark/ml-ciss-behavior/tree/main/ciss-webapp
  * Link to code instructions:

    https://github.com/BU-Spark/ml-ciss-behavior/blob/main/ciss-webapp/Web_App_Directions.md
* Pipeline
  * "Pipeline.ipynb" contains the full pipeline functionality contained in a jupyter notebook. It's primary purpose is to extract features for every second by second frame of video experiments.
  * Link to notebook:

    https://github.com/BU-Spark/ml-ciss-behavior/blob/main/Pipeline/Pipeline.ipynb
  * Link to code instructions:

    https://github.com/BU-Spark/ml-ciss-behavior/blob/main/Pipeline/Pipeline_Directions.md
* Detection Model Training
  * The detection model training notebook is used to retrain a YOLO detection model, to classify the child, parent, and stranger in every video frame
  * Link to notebook:

    https://github.com/BU-Spark/ml-ciss-behavior/blob/main/Pipeline/Detection_Model_Training/YOLO_Retrain.ipynb
  * Link to code instructions:

    https://github.com/BU-Spark/ml-ciss-behavior/blob/main/Pipeline/Detection_Model_Training/Detection_Model_Training_Directions.md
* Feature extraction code
  * This notebook simply shows how we extracted features for each video, passing them through our pipeline.
  * Link to notebook:

    https://github.com/BU-Spark/ml-ciss-behavior/blob/main/Creating_Classifiers/Feature_Extraction/Extracting_video_features.ipynb
  * Code instructions are the same as the Pipeline code
* Label Cleaning
  * The "Client_label_cleaning.ipynb" shows how we transformed the client's labels in a start/stop format to a continuous stream of labels (second by second) that matches our feature extraction format.
  * Link to notebook:

    https://github.com/BU-Spark/ml-ciss-behavior/blob/main/Creating_Classifiers/Cleaned_Labels/Client_label_cleaning.ipynb
  * Link to code instructions:

    https://github.com/BU-Spark/ml-ciss-behavior/blob/main/Creating_Classifiers/Cleaned_Labels/Client_label_cleaning_directions.md
* Decision tree classification
  * The "Decision_Trees.ipynb" shows how we used our extracted features for videos and the client's labels to create decision tree classifiers for future video results.
  * Link to notebook:
  * Link to instructions:

    https://github.com/BU-Spark/ml-ciss-behavior/blob/main/Creating_Classifiers/Decision_Tree_Code_Instructions.md

### Data Sets

* Sharepoint videos
    * BU videos:

      https://bushare.sharepoint.com/sites/GRP-BASE-PEAR-DataCollection/Shared%20Documents/Forms/Item%20View.aspx?id=%2Fsites%2FGRP%2DBASE%2DPEAR%2DDataCollection%2FShared%20Documents%2FData%20Collection%2FBU%20Data%2FVideos%2FBLOCK%202&p=true&ga=1
    * Penn videos:

      https://bushare.sharepoint.com/sites/GRP-BASE-PEAR-DataCollection/Shared%20Documents/Forms/Item%20View.aspx?id=%2Fsites%2FGRP%2DBASE%2DPEAR%2DDataCollection%2FShared%20Documents%2FData%20Collection%2FPenn%20Data%2FVideos%2FBLOCK%202&p=true&ga=1
  * Data for creating decision trees:
    * This data consists of features extracted from the client's videos. We used this data to construct decision trees for classification.

      https://github.com/BU-Spark/ml-ciss-behavior/tree/main/Creating_Classifiers/Decision_Tree_Data
    * Data dictionary:

      https://github.com/BU-Spark/ml-ciss-behavior/blob/main/Creating_Classifiers/Decision_Tree_Data/Decision_Tree_Data_DICT.md
  * Raw client labels
    * This data consists of the researchers manual labels for BU, Penn, and UNC videos. They are in a start/stop format as opposed to a continuous stream. All video labels are self contained on the same spreadsheet.
      
      https://github.com/BU-Spark/ml-ciss-behavior/tree/main/Creating_Classifiers/Cleaned_Labels/RAW_Client_Labels
      * Data dictionary:

        https://github.com/BU-Spark/ml-ciss-behavior/blob/main/Creating_Classifiers/Cleaned_Labels/RAW_Client_Labels/RAW_Labels_DICT.md
  * Cleaned client labels:
    * This data consists of the client's labels for the videos, cleaned to be in the same format as our feature extraction (a continuous label stream second by second). These clean labels were essential to decision tree creation.

      https://github.com/BU-Spark/ml-ciss-behavior/tree/main/Creating_Classifiers/Cleaned_Labels
    * Data dicitonary:
   
      https://github.com/BU-Spark/ml-ciss-behavior/blob/main/Creating_Classifiers/Cleaned_Labels/Cleaned_Labels_DICT.md
  * Manually labeled client frames:
    * This data was used to retrain the yolov8m detection model, to identify the child, parent, and stranger in an image.
    * Here is the path on SCC:
   
      /projectnb/ds549/projects/ciss/Pipeline/YOLO_Labeled_Frames/

### Models
All models are contained within the ciss-webapp folder
Here is the link to all of the models:
https://github.com/BU-Spark/ml-ciss-behavior/tree/main/ciss-webapp/backend/models

* **distance_classifer.pkl:** Decision tree to classify proximity for the parent only. The stranger distance is calculated through logic. (If parent proximity = 2, stranger = 0, same for (1,1) and (0,2))
* **fear_classifer.pkl:** Decision tree to classify fear
* **fear_classifer.pkl:** Decision tree to classify freeze
* **yolo_retrained_model.pt:** Our retrained YOLO detection model to identify the child, parent, and stranger
* **yolov8n-pose.pt:** Pretrained YOLO pose model

## Research
### Relevant Papers

1. Antonov, S. *et al.* (1970) *An intelligent system for video-based proximity analysis*, *SpringerLink*. Available at: [https://link.springer.com/chapter/10.1007/978-981-99-3784-4\_5\#Sec3](https://link.springer.com/chapter/10.1007/978-981-99-3784-4_5#Sec3)  (Accessed: 23 February 2025).   
2. Ji, J., Desai, R. and Niebles, J.C. (no date) *Detecting human-object relationships in videos*. Available at: [https://openaccess.thecvf.com/content/ICCV2021/papers/Ji\_Detecting\_Human-Object\_Relationships\_in\_Videos\_ICCV\_2021\_paper.pdf](https://openaccess.thecvf.com/content/ICCV2021/papers/Ji_Detecting_Human-Object_Relationships_in_Videos_ICCV_2021_paper.pdf?utm_source=chatgpt.com) (Accessed: 23 February 2025).   
3. Conway, A.M. *et al.* (2021) ‘Frame‐by‐frame annotation of video recordings using Deep Neural Networks’, *Ecosphere*, 12(3). doi:10.1002/ecs2.3384. 


Speech Analysis: [https://pmc.ncbi.nlm.nih.gov/articles/PMC7484854/pdf/nihms-1618710.pdf](https://pmc.ncbi.nlm.nih.gov/articles/PMC7484854/pdf/nihms-1618710.pdf)   
“Giving Voice to Vulnerable Children: Machine Learning Analysis of Speech Detects Anxiety and Depression in Early Childhood”

-  Children were asked to create a three-minute story, a task designed to elicit natural speech. Audio recordings of these storytelling sessions were collected for analysis. The recordings were processed using a machine learning algorithm that examined various audio features, including pitch and speech inflections. Specific audio characteristics, such as lower pitch and repetitive speech patterns, were significant indicators of anxiety and depression.

**Interpretability by design using computer vision for behavioral sensing in child and adolescent psychiatry**

 [https://arxiv.org/pdf/2207.04724](https://arxiv.org/pdf/2207.04724) 

The study uses machine learning (ML) and computer vision to automate the detection of behavioral traits in children and adolescents with OCD. Researchers extracted facial expressions, gaze, movement, and vocalization from video recordings of diagnostic interviews. They used pre-trained models such as **OpenFace for gaze tracking** and **YOLOv5 for body movement** analysis. **Motion heatmaps** were generated to quantify movement over time, aiding in the assessment of fear and arousal levels.

Relevant Features for our Project:

- **Facial Expressions:** Use OpenFace or FER (Facial Expression Recognition) models to detect emotional states like fear, anxiety, or engagement.  
- **Body Movement & Freezing Detection:** Implement YOLOv5 for posture estimation and motion heat maps to measure changes in movement, freeze behavior, and avoidance responses.  
- **Proximity Tracking:** Map child’s distance to the stranger and parent over time.  
- In the paper, researchers used the **CIB (Coding Interactive Behavior)** manual, which is a structured system used in psychology to score and quantify behaviors in clinical settings. Instead of a vague "anxiety score" from an AI model, they defined specific human-observable behaviors that experts already use.

#### **Instead of a single AI output, we break it into meaningful behavioral features:**

 **Proximity to the stranger (0-2 scale):** 0 \= Child moves away quickly; 1 \= Child stays at a moderate distance; 2 \= Child approaches or interacts with the stranger

 **Proximity to the parent (0-2 scale):** 0 \= Child stays close to parent throughout; 1 \= Child moves away occasionally but returns; 2 \= Child is independent and does not seek parent

 **Facial Expression Analysis (using OpenFace or FER):** Positive affect (smiling, relaxed face); Neutral (no strong emotions); Negative emotions (fear, sadness, discomfort)

 **Gaze Behavior (tracking face direction);** 1 \= Facing stranger; 2 \= Looking away or avoiding eye contact

 **Body Movement & Freezing (motion analysis using YOLOv5);** 0 \= Very little movement, possibly frozen in fear; 1 \= Normal movement, occasional stillness; 2 \= Active movement, comfortable exploring environment

 **Latency to Respond (reaction time);** How long does it take for the child to react to the stranger?; Delayed responses may indicate **hesitation or fear.**

**Beyond Questionnaires: Video Analysis for Social Anxiety Detection**

[https://arxiv.org/pdf/2501.05461](https://arxiv.org/pdf/2501.05461)

Objective of the study is to detect social anxiety disorder by analyzing behavioral cues captured in video recordings of people giving public speeches, a common stress-induced task. The study examined various bodily indicators, including body pose, head movements, facial features (action units), self-touching, and eye gaze patterns, to differentiate between participants with and without SAD.

Relevant Features for our Project:

- We can extract temporal movement and pose-based features from video footage, just like they extracted speech features.  
- ZCR equivalents for body movements (e.g., rapid head turning or fidgeting) could indicate anxiety.  
- Spectral analysis of **motion heatmaps** could be used to track movement intensity and smoothness.  
- **Temporal segmentation** of behaviors (e.g., before, during, and after a stranger's approach) could enhance interpretability.  
- **Leave-One-Subject-Out (LOSO) Cross-Validation:** The study used LOSO cross-validation to evaluate model performance: Each participant's data was left out once to test model generalization. The model was trained on the rest and tested on the left-out participant.  
- **Davies-Bouldin Index (DBI)** was used for feature selection, reducing the dataset to the eight most predictive features. We can use DBI or similar methods (e.g., PCA, t-SNE) to select the most relevant behavioral features from video (e.g., movement speed, freezing duration, gaze shifts).  
- Used Mann-Whitney U-Tests to ensure observed behavioral differences are not due to chance.


### **Open Source Projects with Documentation**

1. **Object detection: OpenCv/YOLO**

Article: [https://pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/](https://pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/)  
Website: [https://docs.opencv.org/4.11.0/](https://docs.opencv.org/4.11.0/)   
Github: [https://github.com/opencv/opencv](https://github.com/opencv/opencv) 

- OpenCV  
  - Open CV Has hundred of computer vision algorithms  
  - Modular structure, so contains several different libraries  
    - Key libraries for our project:  
      - Video analysis (video)  
        - Includes motion estimation, background subtractions, and object tracking algorithms  
        - Could be useful to track child movement  
    - Has many pre trained DNN models for easy access  
- Object detection (objdetect)  
  - Detection of objects of predefined classes  
  - Tells us where objects are in an image  
  - Utilizes YOLO (you only look once) object detection algorithm  
  - Can use the COCO (common object in context) dataset with many images designed for object detection  
    - Over 330,000 images and more than 1.5 million instances labeled for 80 categories  
  - Could be useful to identify where child is at a given timestep

- Ultralytics/YOLO

Paper: [https://arxiv.org/pdf/1506.02640](https://arxiv.org/pdf/1506.02640)   
Github: [https://github.com/ultralytics/ultralytics/blob/main/docs/en/models/yolov8.md](https://github.com/ultralytics/ultralytics/blob/main/docs/en/models/yolov8.md)

- Yolo Can be used with OpenCv but newer version uses ultralytics  
  - Unified architecture  
    - Treats object detection like a regression problem and directly computes bounding boxes  
      - Processes in one evaluation  
    - Performs in real time  
      - Can handle video streaming with minimal latency  
      - Processes at 45 fps on highly powered GPU  
    - Analyzes the entire image, not just a sliding window. Helps to reduce false positives  
  - For project: Implement with these datasets for extra training  
    [https://universe.roboflow.com/idan-kideckel-67kqi/children-and-adults/dataset/1](https://universe.roboflow.com/idan-kideckel-67kqi/children-and-adults/dataset/1)   
    [https://universe.roboflow.com/a-4euhx/children-vs-adults-yolo-my3ct/dataset/4](https://universe.roboflow.com/a-4euhx/children-vs-adults-yolo-my3ct/dataset/4) 

2. **Object tracking: mmtracking**

Website: [https://mmtracking.readthedocs.io/en/latest/](https://mmtracking.readthedocs.io/en/latest/)   
Github: [https://github.com/open-mmlab/mmtracking?tab=readme-ov-file](https://github.com/open-mmlab/mmtracking?tab=readme-ov-file)

- Open source library built on PyTorch for video tracking models  
  - Video object detection  
  - Single object tracking  
  - Multiple object tracking  
  - Video instance segmentation

3. **Pose Estimation: OpenPose**

Paper: [https://arxiv.org/abs/1812.08008](https://arxiv.org/abs/1812.08008)  
Github: [https://github.com/CMU-Perceptual-Computing-Lab/openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) 

- Open source library built for multi person 2D pose detection, where it identifies key points in the body, feet, hands, and face.   
  - Can be calculated for both photos and videos  
- Part Affinity Fields  
  - PAFs are 2D vector fields that encode the location and orientation of the limbs across an image. OpenPose uses these to associate body parts with individuals effectively  
- Uses a bottom-up approach  
  - Finds all of the body parts in an image first, and then forms them into poses using the PAFs.   
  - The model does not try to detect individuals first

