# Unveiling Deepfakes: A Frequency-Aware Triple Branch Network for Deepfake Detection

![framework](C:\Users\sqh83\Desktop\Unveiling_Deepfake\framework.png)

## Dependencies

    pip install requirement.txt

## Data Preparation

1. Download the original dataset from [FF++](https://github.com/ondyari/FaceForensics).
   
   <!---2. Download the landmark detector from [here](https://github.com/codeniko/shape_predictor_81_face_landmarks).-->

2. Extract frames from FF++ videos. The processed dataset can be downloaded from [FF++](https://pan.baidu.com/s/1ZHm-WCiPjor2Tz2IsuojvA) (code: 7s5s).

3. Run the code in folder *./process* to get the aligned images and masks.

## Results

Our model achieved the following performance on:

| Training Data | Backbone | FF++  | Celeb-DF2 | DFDC_Pre | DFDC  |
| ------------- | -------- | ----- | --------- | -------- | ----- |
| FF++          | Xception | 0.990 | 0.872     | 0.777    | 0.735 |

Note: the metric is *frame-level AUC*.

## Training

To train our model from scratch, please run :

```
    python3  train.py
```
