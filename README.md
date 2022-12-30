# STDAN (Keep Update)

This is the official Pytorch implementation of _STDAN: Deformable Attention Network for Space-Time Video Super-Resolution_.

By [Hai Wang](https://littlewhitesea.github.io/), [Xiaoyu Xiang](https://engineering.purdue.edu/people/xiaoyu.xiang.1), [Yapeng Tian](https://www.yapengtian.com/), [Wenming Yang](http://www.fiesta.tsinghua.edu.cn/pi/3/38), [Qingmin Liao](https://ieeexplore.ieee.org/author/37313219600)

[[Paper]](https://arxiv.org/abs/2203.06841)

## Introduction

### Abstract

The target of space-time video super-resolution (STVSR) is to increase the spatial-temporal resolution of low-resolution (LR) and low frame rate (LFR) videos. Recent approaches based on deep learning have made significant improvements, but most of them only use two adjacent frames, that is, short-term features, to synthesize the missing frame embedding, which cannot fully explore the information flow of consecutive input LR frames. In addition, existing STVSR models hardly exploit the temporal contexts explicitly to assist high-resolution (HR) frame reconstruction. To address these issues, in this paper, we propose a deformable attention network called STDAN for STVSR. First, we devise a long-short term feature interpolation (LSTFI) module, which is capable of excavating abundant content from more neighboring input frames for the interpolation process through a bidirectional RNN structure. Second, we put forward a spatial-temporal deformable feature aggregation (STDFA) module, in which spatial and temporal contexts in dynamic video frames are adaptively captured and aggregated to enhance SR reconstruction.

### LSTFI Module

### STDFA Module



## Dependencies

- Python 3.8.0 (Recommend to use Anaconda)
- PyTorch == 1.8.0
- CUDA == 10.1
- [Deformable Convolution v2](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhu_Deformable_ConvNets_V2_More_Deformable_Better_Results_CVPR_2019_paper.pdf), we use [Detectron2's implementation](https://github.com/facebookresearch/detectron2/tree/main/detectron2/layers/csrc/deformable) in the network.

## Installation

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
   
2. Compile the deformable attention and convolution:
   ```
   cd YOUR_PATH/STDAN/codes/models/modules/DCNv2_latest
   bash make.sh
   ```

## Training

### Dataset preparation

You require to prepare datasets for training the model, the detailed information of Data Preparation is same as [Zooming Slow-Mo](https://github.com/Mukosame/Zooming-Slow-Mo-CVPR-2020#Prerequisites).

### Train the STDAN model

```
cd YOUR_PATH/codes
python train.py -opt options/train/train_stdan.yml
```


## Testing

### Pretrained Models

Our pretrained model can be downloaded via [Google Drive](https://drive.google.com/file/d/1aIbbQYTL2H4F_Uxt2YDY8lxFhfwRPHG4/view?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1S-N5-yujrT4ZnIGc-BFXpA)(access code: gnhk). After you obtain the pretrained model, please put them into the `YOUR_PATH/experiments/pretrained_models` folder.

### Testing on Vid4/SPMC datasets
   
   ```
   cd YOUR_PATH/codes
   python test.py
   ```

### Testing on Vimeo-Slow/Medium/Fast datasets

   ```
   cd YOUR_PATH/codes
   python test_vimeo.py
   ```


## Citation

If you find the code helpful in your research or work, please cite our paper:
```
@article{wang2022stdan,
  title={STDAN: Deformable Attention Network for Space-Time Video Super-Resolution},
  author={Wang, Hai and Xiang, Xiaoyu and Tian, Yapeng and Yang, Wenming and Liao, Qingmin},
  journal={arXiv preprint arXiv:2203.06841},
  year={2022}
}
```
   


## Acknowledgments
We thank [Zooming Slow-Mo](https://github.com/Mukosame/Zooming-Slow-Mo-CVPR-2020) and [Detectron2](https://github.com/facebookresearch/detectron2). They provide many useful codes which facilitate our work.

