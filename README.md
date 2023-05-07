# VT-IQA:Blind Image Quality Index for Authentic Distortions With Local and Global Deep Feature Aggregation
This is the source code for [VT-IQA:Blind Image Quality Index for Authentic Distortions With Local and Global Deep Feature Aggregation](https://ieeexplore.ieee.org/document/9536693).![VT-IQA Framework](https://github.com/esnthere/VT-IQA/blob/main/framework.png)

## Dependencies and Installation
Pytorch: 1.8.1  
timm: 0.3.2  
CUDA: 10.2  

## For test:
This is the simplified  version of published paper  [VT-IQA](https://ieeexplore.ieee.org/document/9536693), where the *Self Attention-Fusion module* is replaced with concatatent operation and the *Sensitivity-based region slection strategy* is replaced with a simple version. This version is much simpler than the orginal version of [VT-IQA](https://ieeexplore.ieee.org/document/9536693), and it achieves nearly the same predicition accuracy and generatlization performance.  This version also have higher computing efficiency. The orginal version of of published paper  [VT-IQA](https://ieeexplore.ieee.org/document/9536693) can be obtained at the 'test_org' folder. 
### 1. Data preparation  
   To ensure high speed, save images and lables of each dataset with 'mat/npz' files. Only need to run '**data_preparation_example_for_koniq.py**' once for each dataset. 
   
### 2. Load pre-trained weight for test  
   The models pre-trained on KonIQ-10k, SPAQ, LIVEW, RBID, CID2013 are released. The dataset are randomly splitted several times during training, and each released model is obtained from one split. The model file '**my_vision_transformer.py**' is modified from open accessed source code of [DEIT](https://github.com/facebookresearch/deit) and [TIMM](https://github.com/huggingface/pytorch-image-models/tree/main/timm). 
   
   The pre-trained models can be downloaded from: [Pre-trained models](https://pan.baidu.com/s/1DGsGBpyTUKPONNwpkxYyWA?pwd=h7ro). Please download these files and put them in the same folder of code and then run '**test_example_of_*dataset*.py**' to test models trained on the *dataset*. You can make both cross-dataset and intra-dataset of the model trained on KonIQ-10k.
   
   
## For train:  
The training code can be available at the 'training' folder.


## If you like this work, please cite:

{   
  author={Li, Leida and Song, Tianshu and Wu, Jinjian and Dong, Weisheng and Qian, Jiansheng and Shi, Guangming},  
  journal={IEEE Transactions on Circuits and Systems for Video Technology},   
  title={Blind Image Quality Index for Authentic Distortions With Local and Global Deep Feature Aggregation},   
  year={2022},  
  volume={32},  
  number={12},  
  pages={8512-8523},  
  doi={10.1109/TCSVT.2021.3112197}    
  }
  
## License
This repository is released under the Apache 2.0 license.  


