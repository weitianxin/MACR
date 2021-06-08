# MACR
This is an implemention for our SIGKDD 2021 paper 

[Model-Agnostic Counterfactual Reasoning for Eliminating Popularity Bias in Recommender System](https://arxiv.org/abs/2010.15363)

by Tianxin Wei, Fuli Feng, Jiawei Chen, Ziwei Wu, Jinfeng Yi and Xiangnan He based on tensorflow.
# Introduction
MACR is a general popularity debias framework based on causal inference and counterfactual reasoning.
# Requirements
tensorflow == 1.14
Numpy
python3
# Datasets
We use several recommendation datasets in the following format:
* train.txt: Biased training data. Each line is user ID, item ID.
* tesst.txt: Unbiased uniform test data. Each line is user ID, item ID.
# Run the code
For example:
```Python
python ./macr_mf/train.py --dataset addressa --batch_size 128 --cuda 1 --pretrain 1 --saveID 1 --log_interval 1 --lr 0.001 --check_c 1 --start -1 --end 1 --step 21 --model_type c
```
# Citation
If you find this paper helpful, please cite our paper.
```
@inproceedings{niu2020counterfactual,
  title={Counterfactual VQA: A Cause-Effect Look at Language Bias},
  author={Niu, Yulei and Tang, Kaihua and Zhang, Hanwang and Lu, Zhiwu and Hua, Xian-Sheng and Wen, Ji-Rong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2021}
}
```





