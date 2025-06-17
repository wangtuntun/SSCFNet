# When Clusters Meet Shape Priors: A Synergistic Framework for Cluster Infrared Small Target Detection

 Official implementation of paper "When Clusters Meet Shape Priors: A Synergistic Framework for Cluster Infrared Small Target Detection". 

# Network Structure

![Backbone](backbone.png)
![Shape Prior Self Evolution module](SPSE.png)
![Convolution Shape Cross Updating module](CSCU.png)
![Dynamic Synergistic Detection module](DSD.png)

# Requirements

* **Python 3.10**
* **Ubuntu20.04 or higher**
* **NVDIA GeForce RTX 4080**
* **Pytorch 1.13.0**
* **More details from requirements.txt**


# Dataset

you can download in [Google Drive](https://drive.usercontent.google.com/download?id=1PY0d1WuCjf_3wAIjDSNhYxREVK27OLzl&export=download&authuser=0) with code of "DenseSIRST".

# Commands for Training

* **Run train.py to train our network**
  ```Run
  Python train.py
  ```


# Weights

We could offer the weights for DenseSIRST [Weight_for_DenseSIRST](best_ckpt_for_DenseSIRST.pth)
