# Rain-PostEFnet

 

## Overview

Accurate precipitation forecasting is essential for
disaster preparedness, water resource management, and climate
modeling. Numerical Weather Prediction (NWP) models, while
extensively utilized, suffer from inherent limitations such as
systematic biases, coarse spatial resolution, and difficulty in ac-
curately predicting extreme rainfall events. Deep learning-based
post-processing has emerged as a promising approach to refine
NWP outputs; however, existing methods struggle with effectively
capturing multi-scale meteorological features and addressing
severe class imbalances in extreme precipitation predictions.
To overcome these challenges, we propose Rain-PostEFNet, a
deep learning framework designed for post-processing NWP
precipitation forecasts. The model integrates EfficientNet for
feature extraction, ECA-Net for adaptive channel attention, and
FPN-Swin-Unet for multi-scale feature refinement, along with a
multi-task learning strategy that jointly optimizes classification
and regression tasks using weighted focal loss and MSE loss. We
evaluate our approach on the PostRainBench datasets (China,
Germany, Korea) using Critical Success Index (CSI), Heidke
Skill Score (HSS), and Accuracy (ACC) as evaluation metrics.
Experimental results demonstrate that Rain-PostEFNet surpasses
state-of-the-art baselines, achieving CSI improvements of 63.94%
(China), 4.28% (Germany), and 2.1% (Korea) for moderate
rain classification, and 78.34% (China), 9.09% (Germany), and
7.68% (Korea) for heavy rain classification. Similarly, HSS scores
improve by 52.18% in China and 2.95% in Germany for rain,
and by 52.4% in China and 6.78% in Germany for heavy rain.
Additionally, an ablation study further confirms the impact of
the multi-task learning framework and attention mechanisms in
refining NWP predictions.

## Dataset

We summarize three NWP datasets with different spatial and temporal resolutions in the following table.

![alt text](./pic/Dataset.png)

## Results

### 🏆 Achieve state-of-the-art in NWP Post-processing based Precipitation Forecasting

Extensive experimental results on the proposed benchmark show that our method outperforms state-of-the-art methods by 6.3%, 4.7%, and 26.8% in rain CSI on the three datasets respectively. Most notably, our model is the first deep learning-based method to outperform traditional Numerical Weather Prediction (NWP) approaches in extreme precipitation conditions. It shows improvements of 15.6%, 17.4%, and 31.8% over NWP predictions in heavy rain CSI on respective datasets. These results highlight the potential impact of our model in reducing the severe consequences of extreme weather events.

![alt text](./pic/Result.png)

### 🌟 Alation Study

We conduct an ablation study by systematically disabling certain components of our CAMT Component and evaluating the CSI results for both rain and heavy rain. Specifically, we focus on the weighted loss, multi-task learning, and channel attention modules as these are unique additions to the Swin-Unet backbone.

In the first part, we use Swin-Unet with CAMT framework (a) as a baseline and we disable each component in CAMT and demonstrate their respective outcomes. In the second part, we use Swin-Unet without CAMT framework (e) as a baseline and we gradually add each component to the model to understand its role.

![alt text](./pic/Ablation.png)

Although Swin-Unet can achieve a relatively high CSI when used alone (e), it does not have the ability to predict heavy rain. Importantly, these three enhancements complement each other. Weighted loss and multi-task learning are effective in improving simultaneous forecasting under the unbalanced distribution of light rain and heavy rain, while CAM provides comprehensive improvements.

## Dataset

Korea Dataset:

https://www.dropbox.com/sh/vbme8g8wtx9pitg/AAAB4o6_GhRq0wMc1JxdXFrVa?dl=0

Germany Dataset：

https://zenodo.org/records/7244319

China Dataset:

https://drive.google.com/file/d/1rBvxtQ8Gh9dXzh-okEOVpA8ZeDzr7yAI/view?usp=drive_link

## Code

Conda Environment

```
conda env create --file enviromental.yml
conda activate PRBench
```

For model training

```
bash scripts/SwinUnet_CAMT.sh
```

Key components of our model are encapsulated in:

- `/model/swinunet_model.py`: Defines the `SwinUnet_CAM_Two` class, incorporating the SwinUnet architecture, multi-task learning heads, and the Channel Attention Module.
- `losses.py`: Including the weighted loss and multi-task learning loss functions.

## Acknowledgement

We appreciate the following GitHub repo very much for the valuable code base and datasets:

https://github.com/osilab-kaist/KoMet-Benchmark-Dataset

https://github.com/DeepRainProject/models_for_radar

https://github.com/HuCaoFighting/Swin-Unet
