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
## MF

Normal MF:
```Python
python ./macr_mf/train.py --dataset addressa --batch_size 1024 --cuda 0 --saveID 1 --log_interval 10 --lr 0.001 --train normalbce --test normal
```
Change the argument dataset to run experiments on different datasets

MACR MF:

```Python
python ./macr_mf/train.py --dataset addressa --batch_size 1024 --cuda 0 --saveID 1 --log_interval 10 --lr 0.001 --train normalbce --test normal
```

Fixing C to 40 can get great performance. Without fixing, tuning the value of C will get higher performance.

MACR MF:

ML10M
```Python
python ./macr_mf/train.py --dataset ml_10m --batch_size 8192 --cuda 0 --saveID 0 --log_interval 10 --lr 0.001 --check_c 1 --start 30 --end 31 --step 1 --train rubibceboth --test rubi --alpha 1e-3 --beta 1e-3
```
![mf_ml10m](https://user-images.githubusercontent.com/37143015/131950971-71a1707f-30bb-4f89-bc3c-03d805414aca.png)

Gowalla
```Python
python ./macr_mf/train.py --dataset gowalla --batch_size 4096 --cuda 0 --saveID 0 --log_interval 10 --lr 0.001 --check_c 1 --start 40 --end 41 --step 1 --train rubibceboth --test rubi --alpha 1e-2 --beta 1e-3
```
![image12](https://user-images.githubusercontent.com/37143015/131951141-fa84d985-e6b3-4d3b-a932-306821504c18.png)

Globe
```Python
python ./macr_mf/train.py --dataset globe --batch_size 4096 --cuda 0 --saveID 0 --log_interval 10 --lr 0.001 --check_c 1 --start 30 --end 31 --step 1 --train rubibceboth --test rubi --alpha 1e-3 --beta 1e-3
```
![image](https://user-images.githubusercontent.com/37143015/131952919-65b9cd9a-87a1-4baf-b9e3-3ce72169cbe3.png)

Yelp2018
```Python
python ./macr_mf/train.py --dataset yelp2018 --batch_size 4096 --cuda 0 --saveID 0 --log_interval 10 --lr 0.001 --check_c 1 --start 40 --end 41 --step 1 --train rubibceboth --test rubi --alpha 1e-2 --beta 1e-3
```
![mf_yelp](https://user-images.githubusercontent.com/37143015/131951078-9a9c4540-55a7-4e08-a329-2578208172b1.png)

Adressa
```Python
python ./macr_mf/train.py --dataset addressa --batch_size 1024 --cuda 0 --saveID 0 --log_interval 10 --lr 0.001 --check_c 1 --start 30 --end 31 --step 1 --train rubibceboth --test rubi --alpha 1e-3 --beta 1e-3
```
![image2](https://user-images.githubusercontent.com/37143015/131950898-27a25c94-a6ee-4194-8d91-054de60ade37.png)
## LightGCN
Normal LightGCN:
```Python
python macr_lightgcn/LightGCN.py --data_path data/ --dataset addressa --verbose 1 --layer_size [64,64] --Ks [20] --loss bce --test normal --epoch 2000 --early_stop 1 --lr 0.001 --batch_size 1024 --gpu_id 1 --log_interval 10
```

MACR LightGCN:

ML10M
```Python
python macr_lightgcn/LightGCN.py --data_path data/ --dataset ml_10m --verbose 1 --layer_size [64,64] --Ks [20] --loss bceboth --test rubiboth --start 40 --end 41 --step 1 --epoch 2000 --early_stop 1 --lr 0.001 --batch_size 8192 --gpu_id 0 --log_interval 10 --alpha 1e-2 --beta 1e-3
```
![image](https://user-images.githubusercontent.com/37143015/131952138-5de9b23b-f12e-432d-9427-3b274580c18c.png)

Gowalla
```Python
python macr_lightgcn/LightGCN.py --data_path data/ --dataset gowalla --verbose 1 --layer_size [64,64] --Ks [20] --loss bceboth --test rubiboth --start 40 --end 41 --step 1 --epoch 2000 --early_stop 1 --lr 0.001 --batch_size 4096 --gpu_id 0 --log_interval 10 --alpha 1e-2 --beta 1e-3
```
![image](https://user-images.githubusercontent.com/37143015/131952287-d04e3a77-ce4f-4bf9-a043-531371bf10e8.png)

Globe
```Python
python macr_lightgcn/LightGCN.py --data_path data/ --dataset globe --verbose 1 --layer_size [64,64] --Ks [20] --loss bceboth --test rubiboth --start 60 --end 61 --step 1 --epoch 2000 --early_stop 1 --lr 0.001 --batch_size 4096 --gpu_id 0 --log_interval 10 --alpha 1e-2 --beta 1e-3
```
![image](https://user-images.githubusercontent.com/37143015/131952397-1de45ac2-f1a5-43a2-9b6d-8cd2634799d2.png)

Yelp2018
```Python
python macr_lightgcn/LightGCN.py --data_path data/ --dataset yelp2018 --verbose 1 --layer_size [64,64] --Ks [20] --loss bceboth --test rubiboth --start 40 --end 41 --step 1 --epoch 2000 --early_stop 1 --lr 0.001 --batch_size 4096 --gpu_id 0 --log_interval 10 --alpha 1e-2 --beta 1e-3
```
![image](https://user-images.githubusercontent.com/37143015/131952545-3fd9a4d8-73d4-418f-8491-a5294cadadec.png)

Adressa
```Python
python macr_lightgcn/LightGCN.py --data_path data/ --dataset addressa --verbose 1 --layer_size [64,64] --Ks [20] --loss bceboth --test rubiboth --start 30 --end 31 --step 1 --epoch 2000 --early_stop 1 --lr 0.001 --batch_size 1024 --gpu_id 0 --log_interval 10 --alpha 1e-2 --beta 1e-3
```
![image](https://user-images.githubusercontent.com/37143015/131951823-2ee91466-e4b9-479b-8e93-06d6c8162e59.png)

The value of counterfactual C can be further fine-grained adjusted for better performance.
# Datasets
We use several recommendation datasets in the following format:
* train.txt: Biased training data. Each line is user ID, item ID.
* test.txt: Unbiased uniform test data. Each line is user ID, item ID.
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






