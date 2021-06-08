import os
import numpy as np
import collections
import json


dataset = 'globe'

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
            if len(items) < K:
                flag = True
                users_to_delete.add(user)
                for item in items:
                    try:
                        while True:
                            item_list[item].remove(user)
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
                        while True:
                            user_list[user].remove(item)
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
        items = [item_dict[i] for i in items]
        temp_user_list[user] = items
    
    for item, users in item_list.items():
        item = item_dict[item]
        users = [user_dict[u] for u in users]
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
                user_list[user].append(item)
                item_list[item].append(user)
            lines = [line]
            pre = line[0]
            preId = line[1]

    lines.sort(key = forth)
    for l in lines:
        user, item, tagID, timestamp = l
        user = int(user) - 1
        item = int(item) - 1
        user_list[user].append(item)
        item_list[item].append(user)

elif dataset == 'addressa':
    f = open('./data/%s/addressa_user_lists.json' % dataset)
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
        for line in new_items:
            user_list[user].append(int(line[0]))
        for line in new_items:
            item_list[int(line[0])].append(user)
elif dataset == 'globe':
    f = open('./data/%s/user_lists.json' % dataset)
    user_lists = eval(f.read())
    for user, items in user_lists.items():
        user = int(user)
        items = [int(i) for i in items]
        user_list[user] = items
        for item in items:
            item_list[item].append(user)

user_list, item_list = filter(user_list, item_list, K = 5)



n_users = len(user_list)
n_items = len(item_list)
n_interactions = 0
for user, items in user_list.items():
    n_interactions += len(items)
print('users:%d\nitems:%d\ninteractions:%d\nsparsity:%.6f' % (n_users, n_items, n_interactions, 1.0*n_interactions/n_items/n_users))
# print(user_list[0])

# for user, items in user_list.items():
#     items = [i + n_users for i in items]
#     user_list[user] = items

with open('./data/%s/user_list.json' % dataset, 'w') as f:
    json.dump(user_list, f)
with open('./data/%s/item_list.json' % dataset, 'w') as f:
    json.dump(item_list, f)


train_user_list = collections.defaultdict(list)
test_user_list = collections.defaultdict(list)
train_item_list = collections.defaultdict(list)
test_item_list = collections.defaultdict(list)

for user, items in user_list.items():
    # print(items)
    splitNum = int(len(items) * 0.8)
    items_list = items
    train_user_list[user] = items_list[:splitNum]
    test_user_list[user] = items_list[splitNum:]
    for item in train_user_list[user]:
        train_item_list[item].append(user)
    for item in test_user_list[user]:
        test_item_list[item].append(user)

cnt = 0
for item in range(n_items):
    if (train_item_list.__contains__(item)):
        if (len(train_item_list[item])==0):
            cnt +=1
    else:
        cnt+=1
print(cnt)


train_inter = 0
test_inter = 0
with open('./data/%s/train1.txt' % dataset, 'w') as f:
    file = ''
    for user, items in train_user_list.items():
    #for user, items in user_list.items():
        train_inter += len(items)
        file += str(user) + ' '
        for item in items:
        #for item in items[:int(0.8*len(items))]:
            file += str(item) + ' '
        
        file = file.strip(' ')
        file += '\n'
    f.write(file)

with open('./data/%s/test1.txt' % dataset, 'w') as f:
    file = ''
    for user, items in test_user_list.items():
    #for user, items in user_list.items():
        test_inter += len(items)
        file += str(user) + ' '
        for item in items:
        #for item in items[int(0.8*len(items)):]:
            file += str(item) + ' '
        file = file.strip(' ')
        file += '\n'
    f.write(file)

print(train_inter/n_users, test_inter/n_users)