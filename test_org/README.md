# Test code for VT-IQA: Blind Image Quality Index for Authentic Distortions With Local and Global Deep Feature Aggregation

## 1. Data preparation  
   To ensure high speed, save images and lables of each dataset with 'mat/npz' files. Only need to run '**data_preparation_example.py**' once for each dataset. 
   
## 2. Load pre-trained weight for test  
   The model pre-trained on KonIQ-10k is released. The dataset are randomly splitted several times during training, and the released model is obtained from the first split. The model file '**my_vision_transformer2.py**' '**my_safusion2.py**' is modified from open accessed source code of [DEIT](https://github.com/facebookresearch/deit) and [TIMM](https://github.com/huggingface/pytorch-image-models/tree/main/timm). 
   
   The pre-trained models can be downloaded from: [Pre-trained models](https://pan.baidu.com/s/1_5YWkppd7bASY7dnIGCdrg?pwd=sz1p). Please download these files and put them in the same folder of code and then run '**test_example_of_koniq_org.py**' to make both cross-dataset and intra-dataset.

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





