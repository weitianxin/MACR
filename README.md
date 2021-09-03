# MACR
This is an implemention for our SIGKDD 2021 paper based on tensorflow

[Model-Agnostic Counterfactual Reasoning for Eliminating Popularity Bias in Recommender System](https://arxiv.org/abs/2010.15363)

by Tianxin Wei, Fuli Feng, Jiawei Chen, Ziwei Wu, Jinfeng Yi and Xiangnan He.
# Introduction
MACR is a general popularity debias framework based on causal inference and counterfactual reasoning.
# Requirements
* tensorflow == 1.14
* Numpy == 1.16.0
* python3.6
* Cython
* CUDA 10
For LightGCN C++ evaluation, please install Cython and do
```Python
cd macr_lightgcn
python setup.py build_ext --inplace
```
# Datasets
We use several recommendation datasets in the following format:
* train.txt: Biased training data. Each line is user ID, item ID.
* tesst.txt: Unbiased uniform test data. Each line is user ID, item ID.
# Run the code
## MF

Normal MF:
```Python
python ./macr_mf/train.py --dataset addressa --batch_size 1024 --cuda 0 --saveID 1 --log_interval 10 --lr 0.001 --train normalbce --test normal
```
Change the argument dataset to run experiments on different datasets

MACR MF:

ML10M
```Python
python ./macr_mf/train.py --dataset ml_10m --batch_size 8192 --cuda 0 --saveID 0 --log_interval 10 --lr 0.001 --check_c 1 --start 30 --end 31 --step 1 --train rubibceboth --test rubi --alpha 1e-3 --beta 1e-3
```
![image](https://user-images.githubusercontent.com/37143015/131950342-94bedf98-b6ef-4728-a819-34ce6a17377b.png)

Gowalla
```Python
python ./macr_mf/train.py --dataset gowalla --batch_size 4096 --cuda 0 --saveID 0 --log_interval 20 --lr 0.001 --check_c 1 --start 40 --end 41 --step 1 --train rubibceboth --test rubi --alpha 1e-2 --beta 1e-3
```
![image](https://user-images.githubusercontent.com/37143015/131949744-3563ad3f-18b6-424e-85ae-25a32f3bc844.png)

Globe
```Python
python ./macr_mf/train.py --dataset globe --batch_size 4096 --cuda 0 --saveID 0 --log_interval 20 --lr 0.001 --check_c 1 --start 30 --end 31 --step 1 --train rubibceboth --test rubi --alpha 1e-3 --beta 1e-3
```
![image](https://user-images.githubusercontent.com/37143015/131950027-5f423d57-a561-4c66-903d-0c5d1d792b73.png)

Yelp2018
```Python
python ./macr_mf/train.py --dataset yelp2018 --batch_size 4096 --cuda 0 --saveID 0 --log_interval 20 --lr 0.001 --check_c 1 --start 40 --end 41 --step 1 --train rubibceboth --test rubi --alpha 1e-2 --beta 1e-3
```
![image](https://user-images.githubusercontent.com/37143015/131950110-07b7e277-ade7-425e-8af9-17956deea5cb.png)

Adressa
```Python
python ./macr_mf/train.py --dataset addressa --batch_size 1024 --cuda 0 --saveID 0 --log_interval 10 --lr 0.001 --check_c 1 --start 30 --end 31 --step 1 --train rubibceboth --test rubi --alpha 1e-3 --beta 1e-3
```
![image](https://user-images.githubusercontent.com/37143015/131950134-e9335b50-5019-4353-a37b-28b909f414b9.png)
## LightGCN
Normal LightGCN:
```Python
python macr_lightgcn/LightGCN.py --data_path data/ --dataset addressa --verbose 1 --layer_size [64,64] --Ks [20] --loss bce --test normal --epoch 2000 --early_stop 1 --lr 0.001 --batch_size 1024 --gpu_id 1 --log_interval 10
```

MACR LightGCN:
```Python
python macr_lightgcn/LightGCN.py --data_path data/ --dataset addressa --verbose 1 --layer_size [64,64] --Ks [20] --loss bceboth --test rubiboth --start 0 --end 50 --step 31 --epoch 2000 --early_stop 1 --lr 0.001 --batch_size 1024 --gpu_id 1 --log_interval 10 --alpha 1e-3 --beta 1e-3
```
(The range of counterfactual C's value needs to be set.)


More details and changes will be added soon.
# Acknowledgement 
Very thanks for Chufeng Shi for his help on code and the [LightGCN](https://github.com/kuandeng/LightGCN) code repo.
# Citation
If you find this paper helpful, please cite our paper.
```
@inproceedings{wei2021model,
  title={Model-Agnostic Counterfactual Reasoning for Eliminating Popularity Bias in Recommender System},
  author={Wei, Tianxin and Feng, Fuli and Chen, Jiawei and Wu, Ziwei and Yi, Jinfeng and He, Xiangnan},
  booktitle={Proceedings of the 27th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  year={2021}
}
```





