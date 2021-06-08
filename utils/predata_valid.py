import os
import numpy as np
import collections
import json
import random as rd
import matplotlib.pyplot as plt
dataset = 'addressa'

def second(elem):
    return elem[1]

def forth(elem):
    return elem[3]

def filter(user_list, item_list, K = 5):
    flag = True
    while flag:
        flag = False
        users_to_delete = set()
        items_to_delete = set()
        for user, items in user_list.items():
            # print(items)
            if len(items) < K:
                flag = True
                users_to_delete.add(user)
                for item in items:
                    try:
                        item_id = item[0]
                        for user_info in item_list[item_id]:
                            if user_info[0] == user:
                                item_list[item_id].remove(user_info)
                        # while True:
                        #     item_list[item[0]].remove(user)
                    except ValueError:
                        continue          
        for user in users_to_delete:
            user_list.pop(user)
            #print('user:', user)
        
        for item, users in item_list.items():
            if len(users) < K:
                flag = True
                items_to_delete.add(item)
                for user in users:
                    try:
                        user_id = user[0]
                        for item_info in user_list[user_id]:
                            if item_info[0] == item:
                                user_list[user_id].remove(item_info)
                        # while True:
                        #     user_list[user[0]].remove(item)
                    except ValueError:
                        continue
        for item in items_to_delete:
            item_list.pop(item)
            #print('item:', item)
    user_dict = dict()
    item_dict = dict()
    user_id = 0
    for user in user_list.keys():
        user_dict[user] = user_id
        user_id += 1
    item_id = 0
    for item in item_list.keys():
        item_dict[item] = item_id
        item_id += 1
    
    temp_user_list = collections.defaultdict(list)
    temp_item_list = collections.defaultdict(list)
    for user, items in user_list.items():
        user = user_dict[user]
        # print(items)
        items = [[item_dict[i[0]], i[1]] for i in items]
        temp_user_list[user] = items
    
    for item, users in item_list.items():
        item = item_dict[item]
        users = [[user_dict[u[0]], u[1]] for u in users]
        temp_item_list[item] = users
    user_list = temp_user_list
    item_list = temp_item_list

    return user_list, item_list

u, n_users, n_items = 0, 0, 0
user_list = collections.defaultdict(list)
item_list = collections.defaultdict(list)
lines = []
pre, preId = 0, 0
if dataset == 'movielens_ml_10m':
    f = open('./data/%s/ratings.dat' % dataset)
    for line in f.readlines():
        line = line.strip('\n').split('::')
        if float(line[2]) < 4.0 :
            continue
        if line[0] == pre or pre == 0:
            lines.append(line)
            pre = line[0]
        else:
            lines.sort(key = forth)
            for l in lines:
                user, item, rating, _ = l
                user = int(user) - 1
                item = int(item) - 1
                user_list[user].append(item)
                item_list[item].append(user)
            lines = [line]
            pre = line[0]


    lines.sort(key = forth)
    for l in lines:
        user, item, rating, _ = l
        user = int(user) - 1
        item = int(item) - 1
        user_list[user].append(item)
        item_list[item].append(user)

elif dataset == 'lastfm':
    f = open('./data/%s/user_taggedartists-timestamps.dat' % dataset)
    # f = open('./LightGCN-master/Data/%s/user_taggedartists-timestamps.dat' % dataset)
    f.readline()
    for line in f.readlines():
        line = line.strip('\n').split('\t')
        # line = line[:-1]
        if line[0] == pre or pre == 0:
            if line[1] == preId:
                lines[-1][-1] = max(lines[-1][-1], line[-1])
                continue
            lines.append(line)
            pre = line[0]
            preId = line[1]
        else:
            lines.sort(key = forth)
            for l in lines:
                user, item, tagID, timestamp = l
                user = int(user) - 1
                item = int(item) - 1
                user_list[user].append([item, int(timestamp)])
                item_list[item].append([user, int(timestamp)])
            lines = [line]
            pre = line[0]
            preId = line[1]

    lines.sort(key = forth)
    for l in lines:
        user, item, tagID, timestamp = l
        user = int(user) - 1
        item = int(item) - 1
        user_list[user].append([item, timestamp])
        item_list[item].append([user, timestamp])

elif dataset == 'addressa':
    f = open('./data/%s/addressa_user_lists.json' % dataset)
    # f = open('./LightGCN-master/Data/%s/addressa_user_lists.json' % dataset)
    user_lists = eval(f.read())
    for user, items in user_lists.items():
        user = int(user)
        timestamp = dict()
        new_items = []
        for item in items:
            item_id = item[0]
            item_stamp = item[1]
            if timestamp.__contains__(item_id):
                timestamp[item_id] = max(timestamp[item_id], item_stamp)
            else:
                timestamp[item_id] = item_stamp
        for item_i, item_s in timestamp.items():
            new_items.append([item_i, item_s])
        new_items.sort(key = second)
        # print(new_items)
        # exit()
        # print(items)
        # print(new_items[0])
        for line in new_items:
            # print(line)
            user_list[user].append([int(line[0]), int(line[1])])
        for line in new_items:
            item_list[int(line[0])].append([user, int(line[1])])
elif dataset == 'globe':
    f = open('./data/%s/user_lists.json' % dataset)
    user_lists = eval(f.read())
    for user, items in user_lists.items():
        user = int(user)
        items = [int(i) for i in items]
        # user_list[user] = items
        user_list[user] = [[items[i],i] for i in range(len(items))]
        id_num = 0
        for item in items:
            item_list[item].append([user,id_num])
            id_num+=1

user_list, item_list = filter(user_list, item_list, K = 5)


max_inters, min_inters = 0, 1e10
for item, users in item_list.items():
    max_inters = max(max_inters, len(users))
    min_inters = min(min_inters, len(users))
print(max_inters, min_inters)



n_users = len(user_list)
n_items = len(item_list)
n_interactions = 0
for user, items in user_list.items():
    n_interactions += len(items)
print('users:%d\nitems:%d\ninteractions:%d\nsparsity:%.6f' % (n_users, n_items, n_interactions, 1.0*n_interactions/n_items/n_users))




# item_list = collections.defaultdict(list)
test_user_list = collections.defaultdict(list)
train_user_list = collections.defaultdict(list)
skew_train_user_list = collections.defaultdict(list)
test_item_list = collections.defaultdict(list)
train_item_list = collections.defaultdict(list)
skew_train_item_list = collections.defaultdict(list)
valid_user_list = collections.defaultdict(list)
valid_item_list = collections.defaultdict(list)








valid_ratio = 0.1

item_list = collections.defaultdict(list)
for user, items in user_list.items():
    split_num = int(len(items) * 0.8)
    for idx in range(len(items)):
        if idx < split_num:
            user_list[user].append(items[idx][0])
        else:
            test_user_list[user].append(items[idx][0])

    for item in items[:split_num]:
        item_list[item[0]].append([user, item[1]])

train_interactions = 0
for user, items in user_list.items():
    train_interactions += len(items)

# balance dataset

# inter_per_item_for_valid = int(train_interactions  / n_items * valid_ratio)
# print(inter_per_item_for_valid)
# for item, users in item_list.items():
#     temp = list(users)
#     temp = sorted(temp, key = lambda x:x[1])
#     temp = [i[0] for i in temp]
#     lim = len(users) - min(int(0.9 * len(users)), inter_per_item_for_valid)
#     train_data = temp[:lim]
#     valid_data = temp[lim:]
#     for user in train_data:
#         train_user_list[user].append(item)
#     for user in valid_data:
#         valid_user_list[user].append(item)

# for user in train_user_list.keys():
#     if len(valid_user_list[user]) == 0:
#         valid_user_list[user].append(train_user_list[user][-1])
#         train_user_list[user] = train_user_list[user][:-1]

# test freq dataset
test_freq = {}
n_test_interactions = 0
for user, items in test_user_list.items():
    n_test_interactions+=len(items)
    for item in items:
        test_item_list[item].append(user)
for item, users in test_item_list.items(): 
    test_freq[item] = len(users)/n_test_interactions
valid_interactions = train_interactions  * valid_ratio
for item, users in item_list.items():
    temp = list(users)
    temp = sorted(temp, key = lambda x:x[1])
    temp = [i[0] for i in temp]
    if item in test_freq:
        freq = test_freq[item]
        lim = len(users) - max(int(freq*valid_interactions),1)
        train_data = temp[:lim]
        valid_data = temp[lim:]
        for user in train_data:
            train_user_list[user].append(item)
        for user in valid_data:
            valid_user_list[user].append(item)
    else:
        train_data = temp
        for user in train_data:
            train_user_list[user].append(item)

print(len(train_user_list), len(valid_user_list), len(test_user_list))


with open('./data/%s/train_freq.txt' % dataset, 'w') as f:
    file = ''
    for user, items in train_user_list.items():
        file += str(user) + ' '
        for item in items:
            file += str(item) + ' '
        
        file = file.strip(' ')
        file += '\n'
    f.write(file)

with open('./data/%s/valid_freq.txt' % dataset, 'w') as f:
    file = ''
    for user, items in valid_user_list.items():
        file += str(user) + ' '
        for item in items:
            file += str(item) + ' '
        
        file = file.strip(' ')
        file += '\n'
    f.write(file)

with open('./data/%s/test.txt' % dataset, 'w') as f:
    file = ''
    for user, items in test_user_list.items():
        file += str(user) + ' '
        for item in items:
            file += str(item) + ' '
        file = file.strip(' ')
        file += '\n'
    f.write(file)