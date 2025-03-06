# Learning Language-guided Adaptive Hyper-modality Representation for Multimodal Sentiment Analysis

Pytorch implementation of paper: 

> [**Learning Language-guided Adaptive Hyper-modality Representation for Multimodal Sentiment Analysis**](https://aclanthology.org/2023.emnlp-main.49.pdf)

> This is a reorganized code, if you find any bugs please contact me. Thanks.


## Content
- [Note](#Note)
- [Data Preparation](#Data-preparation)
- [Environment](#Environment)
- [Training](#Training)
- [Citation](#Citation)


## Note

1. [2025.03.06] The demo code has been updated to fix some issues. We recommend reproducing with new code and environmental requirements.

2. Based on the experience and insights gained from the ALMT, we have futher explored robust MSA by ensuring the integrity of the dominant modality under different noise intensities. This new work has been accepted at NeurIPS 2024, welcome to [this new work](https://github.com/Haoyu-ha/LNLN).

3. The ALMT implementation has been added to [MMSA](https://github.com/thuiar/MMSA); you can also refer to the implementation and make a fairer comparison with other methods in the same framework.

4. We observed that regression metrics (such as MAE and Corr) and classification metrics (such as acc2 and F1) focus on different aspects of model performance. A model that achieves the lowest error in sentiment intensity prediction does not necessarily perform best in classification tasks. To comprehensively demonstrate the capabilities of the model, we selected the best-performing model for each type of metric, meaning that acc2/F1 and MAE correspond to different epochs of the same training process. In addition, the code also compute and report the performance in the same epoch for reference.


## Data Preparation
MOSI/MOSEI/CH-SIMS Download: See [MMSA](https://github.com/thuiar/MMSA).

## Environment
The basic training environment for the results in the paper is Pytorch 2.5.1 with CUDA 12.1, Python 3.11.10 with RTX A40. It should be noted that different hardware and software environments can cause the results to fluctuate.

## Training
You can quickly run the code with the following command:

### CH-SIMS
```
python train.py --config_file configs/sims.yaml --gpu_id 0
```

### MOSI
```
python train.py --config_file configs/mosi.yaml --gpu_id 0
```

### MOSEI
```
python train.py --config_file configs/mosei.yaml --gpu_id 0
```

## Citation

- [Learning Language-guided Adaptive Hyper-modality Representation for Multimodal Sentiment Analysis](https://aclanthology.org/2023.emnlp-main.49/)

Please cite our paper if you find our work useful for your research:

```
@inproceedings{zhang-etal-2023-learning-language,
    title = "Learning Language-guided Adaptive Hyper-modality Representation for Multimodal Sentiment Analysis",
    author = "Zhang, Haoyu  and
              Wang, Yu  and
              Yin, Guanghao  and
              Liu, Kejun  and
              Liu, Yuanyuan  and
              Yu, Tianshu",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    year = "2023",
    publisher = "Association for Computational Linguistics",
    pages = "756--767"
}
```
