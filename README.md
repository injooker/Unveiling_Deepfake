# Unveiling Deepfakes: A Frequency-Aware Triple Branch Network for Deepfake Detection

![framework](https://github.com/injooker/Unveiling_Deepfake/blob/master/framework.png?raw=true)

## Dependencies

    pip install -r requirements.txt

## Data Preparation

1. Download the original dataset from [FF++](https://github.com/ondyari/FaceForensics).
   
   <!---2. Download the landmark detector from [here](https://github.com/codeniko/shape_predictor_81_face_landmarks).-->

2. Extract frames from FF++ videos.

3. Run the code in folder *./process* to get the aligned images and masks.

## Pretrained Model

You can download the pretrained **Xception** weights from here:

- [xception-b5690688.pth](https://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth)  (ImageNet pretrained)

Put the file into the `network/` folder before training or testing.

## Results

Our model achieved the following performance on:

| Training Data | Backbone | FF++  | Celeb-DF2 | Celeb-DF |
| ------------- | -------- | ----- | --------- | -------- |
| FF++          | Xception | 0.990 | 0.872     | 0.796    |

| Training Data | Backbone | Celeb-DF2 |
| ------------- | -------- | --------- | 
| Celeb-DF2     | Xception |   0.999   |

Note: the metric is *frame-level AUC*.

## Training

To train our model from scratch, please run:

```bash
python3 train.py

