# Learning Language-guided Adaptive Hyper-modality Representation for Multimodal Sentiment Analysis

Pytorch implementation of paper: 

> **Learning Language-guided Adaptive Hyper-modality Representation for Multimodal Sentiment Analysis**

> This is a reorganized code, if you find any bugs please contact me. Thanks.


## Content

- [Data Preparation](#Data-preparation)
- [Environment](#Environment)
- [Training](#Training)
- [Test](#Test)
- [Citation](#Citation)


### Note

1. Based on the experience and insights gained from the ALMT, we have futher explored robust MSA by ensuring the integrity of the dominant modality under different noise intensities. This new work has been accepted at NeurIPS 2024, welcome to [this new work](https://github.com/Haoyu-ha/LNLN).

2. The ALMT implementation has been added to [MMSA](https://github.com/thuiar/MMSA); you can also refer to the implementation and make a fair comparison with other methods.

## Data Preparation
MOSI/MOSEI/CH-SIMS Download: See [MMSA](https://github.com/thuiar/MMSA).

## Environment
The basic training environment for the results in the paper is Pytorch 1.11.0, Python 3.7 with RTX 3090. It should be noted that different hardware and software environments can cause the results to fluctuate.

## Training
You can quickly run the code with the following command (you can refer to opts.py to modify more hyperparameters):

### CH-SIMS
```
python train.py --CUDA_VISIBLE_DEVICES 0 --project_name ALMT_DEMO --datasetName sims --dataPath ./datasets/unaligned_39.pkl --fusion_layer_depth 4 --is_test 1
```

### MOSI
```
python train.py --CUDA_VISIBLE_DEVICES 0 --project_name ALMT_DEMO --datasetName mosi --dataPath ./datasets/unaligned_50.pkl --fusion_layer_depth 2 --is_test 1
```

## Test
I provide a trained parameter (the Acc-5 metric for SIMS) for test. You can download it from [Google Drive](https://drive.google.com/file/d/11dYa6mmq7sbgndwe0e_FAtYkYcpjbESo/view?usp=sharing) and [Baidu Netdisk](https://pan.baidu.com/s/1E_is4cOx0DgTlZwPdzHe4g?pwd=659k).

Then, put it to the specified path and run the code with the following command:
```
python test.py --CUDA_VISIBLE_DEVICES 0 --project_name ALMT_Test_SIMS_DEMO --datasetName sims --dataPath ./datasets/unaligned_39.pkl --test_checkpoint ./checkpoint/test/SIMS_Acc5_Best.pth --fusion_layer_depth 4 
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
