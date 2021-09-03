# MACR
This is an implemention for our SIGKDD 2021 paper based on tensorflow

[Model-Agnostic Counterfactual Reasoning for Eliminating Popularity Bias in Recommender System](https://arxiv.org/abs/2010.15363)

by Tianxin Wei, Fuli Feng, Jiawei Chen, Ziwei Wu, Jinfeng Yi and Xiangnan He.
# Introduction
MACR is a general popularity debias framework based on causal inference and counterfactual reasoning.
# Requirements
* tensorflow == 1.14
* Numpy == 1.16.0
* python == 3.6
* Cython == 0.29.24 (Optional)
* CUDA v10

For LightGCN C++ evaluation, please install Cython and do
```Python
cd macr_lightgcn
python setup.py build_ext --inplace
```
# Run the code

Like what we mentioned in our paper, great performance can be obtained by easily setting C=30 or C=40. Here we show the results of setting C=40.

Normal MF :

Change the dataset argument to run experiments on different datasets

```Python
python ./macr_mf/train.py --dataset addressa --batch_size 1024 --cuda 0 --saveID 1 --log_interval 10 --lr 0.001 --train normalbce --test normal
```
MACR MF:

ML10M
```Python
python ./macr_mf/train.py --dataset ml_10m --batch_size 8192 --cuda 0 --saveID 0 --log_interval 10 --lr 0.001 --check_c 1 --c 40 --train rubibceboth --test rubi --alpha 1e-3 --beta 1e-3
```
Gowalla
```Python
python ./macr_mf/train.py --dataset gowalla --batch_size 4096 --cuda 0 --saveID 0 --log_interval 10 --lr 0.001 --check_c 1 --c 40 --train rubibceboth --test rubi --alpha 1e-2 --beta 1e-3
```
Globe
```Python
python ./macr_mf/train.py --dataset globe --batch_size 4096 --cuda 0 --saveID 0 --log_interval 10 --lr 0.001 --check_c 1 --c 40 --train rubibceboth --test rubi --alpha 1e-3 --beta 1e-3
```
Yelp2018
```Python
python ./macr_mf/train.py --dataset yelp2018 --batch_size 4096 --cuda 0 --saveID 0 --log_interval 10 --lr 0.001 --check_c 1 --c 40 --train rubibceboth --test rubi --alpha 1e-2 --beta 1e-3
```
Adressa
```Python
python ./macr_mf/train.py --dataset addressa --batch_size 1024 --cuda 0 --saveID 0 --log_interval 10 --lr 0.001 --check_c 1 --c 40 --train rubibceboth --test rubi --alpha 1e-3 --beta 1e-3
```


Normal LightGCN:

```Python
python macr_lightgcn/LightGCN.py --data_path data/ --dataset addressa --verbose 1 --layer_size [64,64] --Ks [20] --loss bce --test normal --epoch 2000 --early_stop 1 --lr 0.001 --batch_size 1024 --gpu_id 1 --log_interval 10
```

MACR LightGCN:

ML10M
```Python
python macr_lightgcn/LightGCN.py --data_path data/ --dataset ml_10m --verbose 1 --layer_size [64,64] --Ks [20] --loss bceboth --test rubiboth --c 40 --epoch 2000 --early_stop 1 --lr 0.001 --batch_size 8192 --gpu_id 0 --log_interval 10 --alpha 1e-2 --beta 1e-3
```
Gowalla
```Python
python macr_lightgcn/LightGCN.py --data_path data/ --dataset gowalla --verbose 1 --layer_size [64,64] --Ks [20] --loss bceboth --test rubiboth --c 40 --epoch 2000 --early_stop 1 --lr 0.001 --batch_size 4096 --gpu_id 0 --log_interval 10 --alpha 1e-2 --beta 1e-3
```
Globe
```Python
python macr_lightgcn/LightGCN.py --data_path data/ --dataset globe --verbose 1 --layer_size [64,64] --Ks [20] --loss bceboth --test rubiboth --c 40 --epoch 2000 --early_stop 1 --lr 0.001 --batch_size 4096 --gpu_id 0 --log_interval 10 --alpha 1e-2 --beta 1e-3
```
Yelp2018
```Python
python macr_lightgcn/LightGCN.py --data_path data/ --dataset yelp2018 --verbose 1 --layer_size [64,64] --Ks [20] --loss bceboth --test rubiboth --c 40 --epoch 2000 --early_stop 1 --lr 0.001 --batch_size 4096 --gpu_id 0 --log_interval 10 --alpha 1e-2 --beta 1e-3
```
Adressa
```Python
python macr_lightgcn/LightGCN.py --data_path data/ --dataset addressa --verbose 1 --layer_size [64,64] --Ks [20] --loss bceboth --test rubiboth --c 40 --epoch 2000 --early_stop 1 --lr 0.001 --batch_size 1024 --gpu_id 0 --log_interval 10 --alpha 1e-2 --beta 1e-3
```
|                  | **HR**  | Rec     | **NDCG** |
| ---------------- | ------- | ------- | -------- |
| LightGCN_ML10M   | 0.15262 | 0.04745 | 0.02844  |
| LightGCN_Gowalla | 0.28410 | 0.09076 | 0.05999  |
| LightGCN_Globe   | 0.12612 | 0.05271 | 0.02904  |
| LightGCN_Adressa | 0.16456 | 0.12967 | 0.06071  |
| LightGCN_Yelp    | 0.17210 | 0.04016 | 0.02649  |
| MF_ML10M         | 0.14011 | 0.04087 | 0.02407  |
| MF_Gowalla       | 0.26669 | 0.08488 | 0.05756  |
| Mf Globe         | 0.10725 | 0.04594 | 0.02513  |
| MF Adressa       | 0.13561 | 0.10712 | 0.04667  |
| MF Yelp          | 0.14444 | 0.02863 | 0.02039  |



Fixing C to 40 can get great performance. Tuning the value of C through validation will get higher performance. For example on ML10M dataset:

```python
python ./macr_mf/tune.py --dataset ml_10m --batch_size 8192 --cuda 0 --saveID 0 --log_interval 10 --lr 0.001 --check_c 1 --start 30 --end 60 --step 31 --train rubibceboth --test rubi --alpha 1e-3 --beta 1e-3 --valid_set valid
```

|                            | **HR**  | Rec     | **NDCG** |
| -------------------------- | ------- | ------- | -------- |
| MF_ML10M  (C=40)           | 0.14011 | 0.04087 | 0.02407  |
| MF_ML10M (Validation C=32) | 0.14545 | 0.04235 | 0.02518  |

# Acknowledgement 
Very thanks for Chufeng Shi for his help on code and the [LightGCN](https://github.com/kuandeng/LightGCN) code repo.
# Citation
If you find this paper helpful, please cite our paper.
```
@inproceedings{wei2021model,
  title={Model-Agnostic Counterfactual Reasoning for Eliminating Popularity Bias in Recommender System},
  author={Wei, Tianxin and Feng, Fuli and Chen, Jiawei and Wu, Ziwei and Yi, Jinfeng and He, Xiangnan},
  booktitle={Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery \& Data Mining},
  pages={1791--1800},
  year={2021}
}
```





