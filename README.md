# STDAN-CVPRW-2022 (Keep Update)

This is the official Pytorch implementation of _Space-Time Video Super-Resolution Using Deformable Attention Network_.

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

You require to prepare datasets for training the model, the detailed information is elaborated in [Data Preparation](https://github.com/littlewhitesea/STDAN-CVPRW-2022/blob/main/datasets/readme.md).

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


   


## Acknowledgments
We thank [Zooming Slow-Mo](https://github.com/Mukosame/Zooming-Slow-Mo-CVPR-2020) and [Detectron2](https://github.com/facebookresearch/detectron2). They provide many useful codes which facilitate our work.

