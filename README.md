# Causal Graph Transformer

## Introduction
This repository contains the implementation code for paper:

**Causal Graph Transformer for Treatment Effect Estimation Under Unknown Interference** 

Anpeng Wu, Haiyi Qiu, Zhengming Chen, Zijian Li, Ruoxuan Xiong, Fei Wu, and Kun Zhang

<[https://arxiv.org/abs/2006.07040](https://openreview.net/forum?id=foQ4AeEGG7)>

## Data Availability

BlogCatalog (BC) is an online community that offers blog services, where each blogger represents a study unit within the dataset. The relationships between these units form the social network, with each edge denoting a social link. The features are bag-of-words representations of keywords in the bloggers’ descriptions.

Flickr is an online social network providing image and video sharing services. The dataset is constructed by forming links between images that share common metadata. In this dataset, each instance is a user, and each edge represents the social relationship between two users. The features of each user represent a list of tags indicating their interests. 

The BlogCatalog and Flickr datasets are available at https://github.com/songjiang0909/Causal-Inference-on-Networked-Data.

## Configuration

```shell
conda create -n py39 python=3.9
pip install numpy==1.26.4 scipy==1.13.0 pandas==2.2.2 torch==2.3.0 scikit-learn==1.4.2 openpyxl==3.1.2 torch_geometric==2.5.2 torch-scatter==1.1.0
```

Hardware used: (1) MacBook Pro with Apple M2 Pro. (2) Ubuntu 16.04.3 LTS operating system with 2 * Intel Xeon E5-2660 v3 @ 2.60GHz CPU (40 CPU cores, 10 cores per physical CPU, 2 threads per core), 256 GB of RAM, and 4 * GeForce GTX TITAN X GPU with 12GB of VRAM. 

Software used: Python 3.9 with numpy 1.26.4, scipy 1.13.0, pandas 2.2.2, torch 2.3.0, scikit-learn 1.4.2, openpyxl 3.1.2, torch\_geometric 2.5.2, torch-scatter 1.1.0.

