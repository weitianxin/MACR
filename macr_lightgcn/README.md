# LightGCN
This is our Tensorflow implementation for the paper:

>Xiangnan He, Kuan Deng ,Xiang Wang, Yan Li, Yongdong Zhang, Meng Wang(2020). LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation, [Paper in arXiv](https://arxiv.org/abs/2002.02126).

Contributors: Dr. Xiangnan He (staff.ustc.edu.cn/~hexn/), Kuan Deng, Yingxin Wu.

## Introduction
In this work, we aim to simplify the design of GCN to make it more concise and appropriate for recommendation. We propose a new model named LightGCN, including only the most essential component in GCN—neighborhood aggregation—for collaborative filtering

## Environment Requirement
The code has been tested running under Python 3.6.5. The required packages are as follows:
* tensorflow == 1.14.0
* numpy == 1.14.3
* scipy == 1.1.0
* sklearn == 0.19.1
* cython == 0.29.15
## C++ evaluator
We have implemented C++ code to output metrics during and after training, which is much more efficient than python evaluator. It needs to be compiled first using the following command. 
```
python setup.py build_ext --inplace
```
After compilation, the C++ code will run by default instead of Python code.

## Examples
The instruction of commands has been clearly stated in the codes (see the parser function in LightGCN/utility/parser.py).

~/anaconda3/bin/python LightGCN.py --dataset kdd2020 --regs [1e-4] --embed_size 64 --layer_size [64,64,64] --lr 0.001 --batch_size 2048 --epoch 1000 --gpu_id 0

~/anaconda3/bin/python LightGCN.py --dataset lastfm --regs [1e-4] --embed_size 64 --layer_size [64,64,64] --lr 0.001 --batch_size 2048 --epoch 1000 --gpu_id 0

~/anaconda3/bin/python LightGCN.py --dataset lastfm_skew --regs [1e-4] --embed_size 64 --layer_size [64,64,64] --lr 0.001 --batch_size 2048 --epoch 1000 --gpu_id 0

~/anaconda3/bin/python LightGCN.py --dataset addressa --regs [1e-4] --embed_size 64 --layer_size [64,64,64] --lr 0.001 --batch_size 2048 --epoch 1000 --gpu_id 2

~/anaconda3/bin/python LightGCN.py --dataset ml10m --regs [1e-4] --embed_size 64 --layer_size [64,64,64] --lr 0.001 --batch_size 4096 --epoch 1000  --gpu_id 0

=======
