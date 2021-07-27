# MACR
This is an implemention for our SIGKDD 2021 paper based on tensorflow

[Model-Agnostic Counterfactual Reasoning for Eliminating Popularity Bias in Recommender System](https://arxiv.org/abs/2010.15363)

by Tianxin Wei, Fuli Feng, Jiawei Chen, Ziwei Wu, Jinfeng Yi and Xiangnan He.
# Introduction
MACR is a general popularity debias framework based on causal inference and counterfactual reasoning.
# Requirements
* tensorflow == 1.14
* Numpy
* python3
* Cython
# Datasets
We use several recommendation datasets in the following format:
* train.txt: Biased training data. Each line is user ID, item ID.
* tesst.txt: Unbiased uniform test data. Each line is user ID, item ID.
# Run the code
For example:\\
Normal MF:
```Python
python ./macr_mf/train.py --dataset addressa --batch_size 128 --cuda 1 --saveID 1 --log_interval 10 --lr 0.001 --train normalbce --test normal
```
MACR MF:
```Python
python ./macr_mf/train.py --dataset addressa --batch_size 128 --cuda 1 --saveID 1 --log_interval 1 --lr 0.001 --check_c 1 --start -1 --end 1 --step 21 --train rubibceboth --test rubi --alpha 1e-3 --beta 1e-3
```

Normal LightGCN:
```Python
python macr_lightgcn/LightGCN.py --data_path data/ --dataset addressa --verbose 1 --layer_size [64,64] --Ks [20] --loss bce --test normal --epoch 2000 --early_stop 1 --lr 0.001 --batch_size 1024 --gpu_id 1 --log_interval 10
```
MACR LightGCN:
```Python
python macr_lightgcn/LightGCN.py --data_path data/ --dataset addressa --verbose 1 --layer_size [64,64] --Ks [20] --loss bceboth --test rubiboth --start 0 --end 50 --step 31 --epoch 2000 --early_stop 1 --lr 0.001 --batch_size 1024 --gpu_id 1 --log_interval 10 --alpha 1e-3 --beta 1e-3
```
(The range of counterfactual C's value needs to be set.)\\
More details and changes will be added soon.
# Acknowledgement 
Very thanks for Chufeng Shi for his help on code and the LightGCN code repo.
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





