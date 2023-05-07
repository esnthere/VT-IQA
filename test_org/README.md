# Training Code for IEIT: Blind Image Quality Assessment for Authentic Distortions by Intermediary Enhancement and Iterative Training
This is the simplified training example of VT-IQA on the KonIQ-10k dataset. The trainning process is the same for other datasets:

## 1. Data preparation

   To ensure high speed, save training images and lables, enhanced images, probability for image selection into 'mat/npz' files. Please run '**data_preparation_example_for_koniq.py**' to save the training images and labels, and other necessary files can be downloaded from [Trainng files](https://pan.baidu.com/s/14KLIdtwTvmqQvppUv181Vw?pwd=27z7). Please download these files and put them into the same folder of the training code.
   
## 2. Training the model

   Please run '**training_example_of_koniq.py**' to performe the training process.    
   The training example of the first and second stage can be seen from '**run training_example_of_koniq.ipynb**'(part results).  
   The pre-trained weight and model file '**my_vision_transformer.py**' is modified from open accessed source code of [DEIT](https://github.com/facebookresearch/deit) and [TIMM](https://github.com/huggingface/pytorch-image-models/tree/main/timm). 

## If you like this work, please cite:

{
  author={Song, Tianshu and Li, Leida and Chen, Pengfei and Liu, Hantao and Qian, Jiansheng},  
  journal={IEEE Transactions on Circuits and Systems for Video Technology},   
  title={Blind Image Quality Assessment for Authentic Distortions by Intermediary Enhancement and Iterative Training},   
  year={2022},  
  volume={32},  
  number={11},  
  pages={7592-7604},  
  doi={10.1109/TCSVT.2022.3179744}   
}




