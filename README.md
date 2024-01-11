# Learning Language-guided Adaptive Hyper-modality Representation for Multimodal Sentiment Analysis

Pytorch implementation of paper: 

> **Learning Language-guided Adaptive Hyper-modality Representation for Multimodal Sentiment Analysis**

## Content

- [Data Preparation](#data-preparation)
- [Training](#Training)
- [Citation](#citation)
  
## Data Preparation

MOSI/MOSEI/CH-SIMS Download: See [MMSA](https://github.com/thuiar/MMSA)

## Training
You can quickly run the code with the following command (you can refer to opt.py to modify more hyperparameters):
### MOSI
```
python train.py --CUDA_VISIBLE_DEVICES 0 --project_name ALMT_DEMO --datasetName mosi --dataPath ./datasets/unaligned_50.pkl --fusion_layer_depth 2 --is_test 1
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
