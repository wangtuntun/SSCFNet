# When Clusters Meet Shape Priors: A Synergistic Framework for Cluster Infrared Small Target Detection [[Paper]](https://ieeexplore.ieee.org/abstract/document/11164479) 

Tuntun Wang, Jincheng Zhou, Yuxin Jing, Tianpei Zhang, IEEE Transactions on Geoscience and Remote Sensing 2025.

# If the implementation of this repo is helpful to you, just star it！⭐⭐⭐

# Chanlleges and inspiration   
![Image text](https://github.com/wangtuntun/SSCFNet/blob/master/fig2_intro.png)

# Structure
![Image text](https://github.com/wangtuntun/SSCFNet/blob/master/fig3_framework.png)

![Image text](https://github.com/wangtuntun/SSCFNet/blob/master/fig4_SPSE.png)

![Image text](https://github.com/wangtuntun/SSCFNet/blob/master/fig5_CSCU.png)

![Image text](https://github.com/wangtuntun/SSCFNet/blob/master/fig6_DSD.png)

# Introduction

We propose the Synergistic Shape-Contextual Fusion Network (SSCFNet) to the cluster IRSTD task. Experiments on DenseSIRST demonstrate the effectiveness of our method. Our main contributions are as follows:

1. We analyze the challenges posed by small infrared targets in dense clustered scenarios and propose SPSE to generate a global shape prior that explicitly encodes positional constraints.

2. We propose the CSCU dynamically fuses local convolutional features with global shape priors, aiming to mitigate the convergence challenges of implicit self-attention-based shape enhancement.

3. We design the DSD, which dynamically integrates convolutional features of individual targets with cluster-level shape constraints to achieve adaptive target detection under varying background complexities.


## Usage

#### 1. Data

you can download in [Google Drive](https://drive.usercontent.google.com/download?id=1PY0d1WuCjf_3wAIjDSNhYxREVK27OLzl&export=download&authuser=0) with code of "DenseSIRST".


##### 2. Train.
```bash
python train.py
```

#### 3. Test and demo.
weight：https://github.com/wangtuntun/SSCFNet/blob/master/best_ckpt_for_DenseSIRST.pth
```bash
python test.py
```

## Results and Trained Models

#### Visual Results
![Image text](https://github.com/wangtuntun/SSCFNet/blob/master/fig7_vis_compare.png)

![Image text](https://github.com/wangtuntun/SSCFNet/blob/master/fig9_vis_compare_3D.png)

![Image text](https://github.com/wangtuntun/SSCFNet/blob/master/fig10_vis_ablation_3D.png)


*The code and overall repository style is highly borrowed from [SCTransNet](https://github.com/xdFai/SCTransNet). Thanks to Shuai Yuan.

## Citation

If you find the code useful, please consider citing our paper using the following BibTeX entry.

```
@ARTICLE{10486932,
  author={Wang, Tuntun and Zhou, Jincheng and Jing, Yuxin and Zhang, Tianpei},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={When Clusters Meet Shape Priors: A Synergistic Framework for Cluster Infrared Small Target Detection}, 
  year={2025},
  volume={63},
  number={},
  pages={1-15},
  keywords={Shape;Feature extraction;Deep learning;Object detection;Noise;Clutter;Spatial resolution;Filters;Data mining;Training;Clustered target;infrared small target detection (IRSTD);shape prior;synergistic shape-contextual fusion},
  doi={10.1109/TGRS.2025.3610139}}
```


## Contact
**Welcome to raise issues or email to [carlwang@smgtu.edu.cn](carlwang@smgtu.edu.cn) for any question regarding our SSCFNet.**
