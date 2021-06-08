import os
import numpy as np
import collections
import json
import random as rd
import matplotlib.pyplot as plt
dataset = 'yelp2018'

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
                user_list[user].append([item, _])
                item_list[item].append([user, _])
            lines = [line]
            pre = line[0]


    lines.sort(key = forth)
    for l in lines:
        user, item, rating, _ = l
        user = int(user) - 1
        item = int(item) - 1
        user_list[user].append([item, _])
        item_list[item].append([user, _])

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

elif dataset == 'gowalla':
    train_file = './data/gowalla/train_ori.txt'
    test_file = './data/gowalla/test_ori.txt'

    count = dict()
    with open(train_file) as f:
        for line in f.readlines():
            line = line.strip('\n').split(' ')
            if len(line) == 0:
                continue
            line = [int(i) for i in line]
            user = line[0]
            items = line[1:]
            if (len(items)==0):
                continue

            count[user] = 0
            id_num = 0
            for item in items:
                user_list[user].append([item, id_num])
                item_list[item].append([user, id_num])
                id_num += 1
            count[user] = id_num

    with open(test_file) as f:
        for line in f.readlines():
            line = line.strip('\n').split(' ')
            if len(line) == 0:
                continue
            line = [int(i) for i in line]
            user = line[0]
            items = line[1:]
            if len(items) == 0:
                continue
            id_num = count[user]
            for item in items:
                user_list[user].append([item, id_num])
                item_list[item].append([user, id_num])
                id_num += 1
            count[user] = id_num
elif dataset == 'yelp2018':
    train_file = './data/yelp2018/train_ori.txt'
    test_file = './data/yelp2018/test_ori.txt'

    count = dict()
    with open(train_file) as f:
        for line in f.readlines():
            line = line.strip('\n').split(' ')
            if len(line) == 0:
                continue
            line = [int(i) for i in line]
            user = line[0]
            items = line[1:]
            if (len(items)==0):
                continue

            count[user] = 0
            id_num = 0
            for item in items:
                user_list[user].append([item, id_num])
                item_list[item].append([user, id_num])
                id_num += 1
            count[user] = id_num

    with open(test_file) as f:
        for line in f.readlines():
            line = line.strip('\n').split(' ')
            if len(line) == 0:
                continue
            line = [int(i) for i in line]
            user = line[0]
            items = line[1:]
            if len(items) == 0:
                continue
            id_num = count[user]
            for item in items:
                user_list[user].append([item, id_num])
                item_list[item].append([user, id_num])
                id_num += 1
            count[user] = id_num
            

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







# balance dataset
unbias_ratio = 0.2
# inter_per_item_for_unbias = int(n_interactions/n_items*unbias_ratio)
inter_per_item_for_unbias = 5
# print(inter_per_item_for_unbias)

for item, users in item_list.items():
    temp = list(users)
    temp = sorted(temp, key = lambda x:x[1])
    temp = [i[0] for i in temp]
    lim = len(users) - min(min(int(0.9 * len(users)), len(users)-1), inter_per_item_for_unbias)
    bias_data = temp[:lim]
    unbias_data = temp[lim:]
    for user in bias_data:
        train_user_list[user].append(item)
    
    # valid_lim = int(len(unbias_data)/2.0)
    # for user in unbias_data[:valid_lim]:
    #     valid_user_list[user].append(item)
    # for user in unbias_data[valid_lim:]:
    #     test_user_list[user].append(item)
    for user in unbias_data:
        test_user_list[user].append(item)

# for user in train_user_list.keys():
#     if len(valid_user_list[user]) == 0:
#         valid_user_list[user].append(train_user_list[user][-1])
#         train_user_list[user] = train_user_list[user][:-1]

print(len(train_user_list), len(valid_user_list), len(test_user_list))

for user, items in train_user_list.items():
    for item in items:
        train_item_list[item].append(user)

for user, items in test_user_list.items():
    for item in items:
        test_item_list[item].append(user)

x = list()
y = list()
for user, items in train_user_list.items():
    y.append(len(items))
x = list(range(len(train_user_list.keys())))
plt.bar(x, y)
plt.yticks([])
plt.savefig('./figures/%s_train_user.png' % dataset)
plt.cla()

x = list()
y = list()
for user, items in test_user_list.items():
    y.append(len(items))
x = list(range(len(test_user_list.keys())))
plt.bar(x, y)
plt.yticks([])
plt.savefig('./figures/%s_test_user.png' % dataset)
plt.cla()

x = list()
y = list()
for item, users in train_item_list.items():
    y.append(len(users))
x = list(range(len(train_item_list.keys())))
plt.bar(x, y)
plt.yticks([])
plt.savefig('./figures/%s_train_item.png' % dataset)
plt.cla()

x = list()
y = list()
for item, users in test_item_list.items():
    y.append(len(users))
x = list(range(len(test_item_list.keys())))
plt.bar(x, y)
plt.yticks([])
plt.savefig('./figures/%s_test_item.png' % dataset)


with open('./data/%s/train.txt' % dataset, 'w') as f:
    file = ''
    for user, items in train_user_list.items():
        file += str(user) + ' '
        for item in items:
            file += str(item) + ' '
        
        file = file.strip(' ')
        file += '\n'
    f.write(file)

with open('./data/%s/valid.txt' % dataset, 'w') as f:
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