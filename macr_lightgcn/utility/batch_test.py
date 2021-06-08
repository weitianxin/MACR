'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
from utility.parser import parse_args
from utility.load_data import *
from evaluator import eval_score_matrix_foldout
import multiprocessing
import os
import heapq
import numpy as np
cores = multiprocessing.cpu_count() // 2

args = parse_args()
data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size)
# data_generator.check()
USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test

BATCH_SIZE = args.batch_size


def test(sess, model, users_to_test, drop_flag=False, train_set_flag=0, method="normal"):
    # data_generator.check()
    # B: batch size
    # N: the number of items
    top_show = np.sort(model.Ks)
    max_top = max(top_show)
    result = {'hr': np.zeros(len(model.Ks)), 'recall': np.zeros(len(model.Ks)), 'ndcg': np.zeros(len(model.Ks))}

    u_batch_size = BATCH_SIZE

    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1
    
    count = 0
    all_result = []
    item_batch = range(ITEM_NUM)
    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]
        if method=="normal":
            if drop_flag == False:
                rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
                                                            model.pos_items: item_batch})
            else:
                rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
                                                            model.pos_items: item_batch,
                                                            model.node_dropout: [0.] * len(eval(args.layer_size)),
                                                            model.mess_dropout: [0.] * len(eval(args.layer_size))})
        elif method == 'causal':
            if drop_flag == False:
                rate_batch = sess.run(model.batch_ratings_causal_c, {model.users: user_batch,
                                                                model.pos_items: item_batch})
            else:
                rate_batch = sess.run(model.batch_ratings_causal_c, {model.users: user_batch,
                                                                model.pos_items: item_batch,
                                                                model.node_dropout: [0.] * len(eval(args.layer_size)),
                                                                model.mess_dropout: [0.] * len(eval(args.layer_size))})
        elif method == 'rubi1':
            if drop_flag == False:
                rate_batch = sess.run(model.rubi_ratings1, {model.users: user_batch,
                                                                model.pos_items: item_batch})
            else:
                rate_batch = sess.run(model.rubi_ratings1, {model.users: user_batch,
                                                                model.pos_items: item_batch,
                                                                model.node_dropout: [0.] * len(eval(args.layer_size)),
                                                                model.mess_dropout: [0.] * len(eval(args.layer_size))})
        elif method == 'rubi2':
            if drop_flag == False:
                rate_batch = sess.run(model.rubi_ratings2, {model.users: user_batch,
                                                                model.pos_items: item_batch})
            else:
                rate_batch = sess.run(model.rubi_ratings2, {model.users: user_batch,
                                                                model.pos_items: item_batch,
                                                                model.node_dropout: [0.] * len(eval(args.layer_size)),
                                                                model.mess_dropout: [0.] * len(eval(args.layer_size))})

        elif method == 'rubiboth':
            if drop_flag == False:
                rate_batch = sess.run(model.rubi_ratings_both, {model.users: user_batch,
                                                                model.pos_items: item_batch})
            else:
                rate_batch = sess.run(model.rubi_ratings_both, {model.users: user_batch,
                                                                model.pos_items: item_batch,
                                                                model.node_dropout: [0.] * len(eval(args.layer_size)),
                                                                model.mess_dropout: [0.] * len(eval(args.layer_size))})
        item_acc_list = {}
        rate_batch = np.array(rate_batch)# (B, N)
        for i in range(data_generator.n_items):
            item_acc_list[i] = 0
        all_items = set(range(data_generator.n_items))
        for i, rate_user in enumerate(rate_batch):
            user = user_batch[i]
            user_pos_test = data_generator.test_set[user]
            try:
                train_items = data_generator.train_items[user]
            except:
                train_items = []
            test_items = list(all_items - set(train_items))
            item_score = dict()
            for i in test_items:
                item_score[i] = rate_user[i]
            K_max_item_score = heapq.nlargest(20, item_score, key = item_score.get)
            for i in K_max_item_score:
                if i in user_pos_test:
                    item_acc_list[i] += 1/len(data_generator.test_item_set[i])
        with open("Lightgcn_macr.txt","w") as f:
            f.write(str(item_acc_list))
        exit()
        test_items = []
        if train_set_flag == 0:
            for user in user_batch:
                set_list = data_generator.test_set[user]
                test_items.append(set_list)# (B, #test_items)
                
            # set the ranking scores of training items to -inf,
            # then the training items will be sorted at the end of the ranking list.    
            for idx, user in enumerate(user_batch):
                if user in data_generator.train_items.keys():
                    train_items_off = data_generator.train_items[user]
                else:
                    train_items_off = []
                rate_batch[idx][train_items_off] = -np.inf
        else:
            for user in user_batch:
                # test_items.append(data_generator.train_items[user])
                test_items.append(data_generator.test_set[user])
        batch_result = eval_score_matrix_foldout(rate_batch, test_items, max_top)#(B,k*metric_num), max_top= 20
        count += len(batch_result)
        all_result.append(batch_result)
        
    
    assert count == n_test_users
    all_result = np.concatenate(all_result, axis=0)
    # print(all_result.shape)
    # x, y = all_result.shape
    for i in range(all_result.shape[0]):
        for j in range(2*max_top, 3*max_top):
            # print(all_result[i][j-1])
            if all_result[i][j-max_top] != 0:
                all_result[i][j] = 1.0
            else:
                all_result[i][j] = 0.0
    # print(all_result)
    final_result = np.mean(all_result, axis=0)  # mean
    # print(final_result)
    final_result = np.reshape(final_result, newshape=[5, max_top])
    # print(final_result)
    final_result = final_result[:, top_show-1]
    # print(final_result)
    final_result = np.reshape(final_result, newshape=[5, len(top_show)])
    # print(final_result)
    result['hr'] += final_result[2]
    result['recall'] += final_result[1]
    result['ndcg'] += final_result[3]
    return result
               
            








