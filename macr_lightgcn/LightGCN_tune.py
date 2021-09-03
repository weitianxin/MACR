'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)

version:
Parallelized sampling on CPU
C++ evaluation for top-k recommendation
'''

import os
import sys
import threading
import tensorflow as tf
import random
import logging
from tensorflow.python.client import device_lib
from utility.helper import *
from utility.batch_test import *
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
logging.getLogger().setLevel(logging.INFO)

gpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
cpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'CPU']

class LightGCN(object):
    def __init__(self, data_config, pretrain_data):
        # argument settings
        self.model_type = 'LightGCN'
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type
        self.pretrain_data = pretrain_data
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_fold = 100
        self.norm_adj = data_config['norm_adj']
        self.n_nonzero_elems = self.norm_adj.count_nonzero()
        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        self.log_dir=self.create_model_str()
        self.verbose = args.verbose
        self.Ks = eval(args.Ks)


        '''
        *********************************************************
        Create Placeholder for Input Data & Dropout.
        '''
        # placeholder definition
        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))
        
        self.node_dropout_flag = args.node_dropout_flag
        self.node_dropout = tf.placeholder(tf.float32, shape=[None])
        self.mess_dropout = tf.placeholder(tf.float32, shape=[None])
        with tf.name_scope('TRAIN_LOSS'):
            self.train_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('train_loss', self.train_loss)
            self.train_mf_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('train_mf_loss', self.train_mf_loss)
            self.train_emb_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('train_emb_loss', self.train_emb_loss)
            self.train_reg_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('train_reg_loss', self.train_reg_loss)
        self.merged_train_loss = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, 'TRAIN_LOSS'))
        
        
        with tf.name_scope('TRAIN_ACC'):
            self.train_rec_first = tf.placeholder(tf.float32)
            #record for top(Ks[0])
            tf.summary.scalar('train_rec_first', self.train_rec_first)
            self.train_rec_last = tf.placeholder(tf.float32)
            #record for top(Ks[-1])
            tf.summary.scalar('train_rec_last', self.train_rec_last)
            self.train_ndcg_first = tf.placeholder(tf.float32)
            tf.summary.scalar('train_ndcg_first', self.train_ndcg_first)
            self.train_ndcg_last = tf.placeholder(tf.float32)
            tf.summary.scalar('train_ndcg_last', self.train_ndcg_last)
        self.merged_train_acc = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, 'TRAIN_ACC'))

        with tf.name_scope('TEST_LOSS'):
            self.test_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('test_loss', self.test_loss)
            self.test_mf_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('test_mf_loss', self.test_mf_loss)
            self.test_emb_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('test_emb_loss', self.test_emb_loss)
            self.test_reg_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('test_reg_loss', self.test_reg_loss)
        self.merged_test_loss = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, 'TEST_LOSS'))

        with tf.name_scope('TEST_ACC'):
            self.test_rec_first = tf.placeholder(tf.float32)
            tf.summary.scalar('test_rec_first', self.test_rec_first)
            self.test_rec_last = tf.placeholder(tf.float32)
            tf.summary.scalar('test_rec_last', self.test_rec_last)
            self.test_ndcg_first = tf.placeholder(tf.float32)
            tf.summary.scalar('test_ndcg_first', self.test_ndcg_first)
            self.test_ndcg_last = tf.placeholder(tf.float32)
            tf.summary.scalar('test_ndcg_last', self.test_ndcg_last)
        self.merged_test_acc = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, 'TEST_ACC'))
        """
        *********************************************************
        Create Model Parameters (i.e., Initialize Weights).
        """
        # initialization of model parameters
        self.weights = self._init_weights()

        """
        *********************************************************
        Compute Graph-based Representations of all users & items via Message-Passing Mechanism of Graph Neural Networks.
        Different Convolutional Layers:
            1. ngcf: defined in 'Neural Graph Collaborative Filtering', SIGIR2019;
            2. gcn:  defined in 'Semi-Supervised Classification with Graph Convolutional Networks', ICLR2018;
            3. gcmc: defined in 'Graph Convolutional Matrix Completion', KDD2018;
        """
        if self.alg_type in ['lightgcn']:
            self.ua_embeddings, self.ia_embeddings = self._create_lightgcn_embed()
            
        elif self.alg_type in ['ngcf']:
            self.ua_embeddings, self.ia_embeddings = self._create_ngcf_embed()

        elif self.alg_type in ['gcn']:
            self.ua_embeddings, self.ia_embeddings = self._create_gcn_embed()

        elif self.alg_type in ['gcmc']:
            self.ua_embeddings, self.ia_embeddings = self._create_gcmc_embed()

        """
        *********************************************************
        Establish the final representations for user-item pairs in batch.
        """
        self.u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.users)
        self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.pos_items)
        self.neg_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.neg_items)
        self.u_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['user_embedding'], self.users)
        self.pos_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['item_embedding'], self.pos_items)
        self.neg_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['item_embedding'], self.neg_items)

        """
        *********************************************************
        Establish 2 brach.
        """
        self.alpha = args.alpha
        self.beta = args.beta
        self.rubi_c = tf.Variable(tf.zeros([1]), name = 'rubi_c')
        self.sigmoid_yu = tf.squeeze(tf.nn.sigmoid(tf.matmul(self.weights['user_embedding'], self.w_user)))
        self.sigmoid_yi = tf.squeeze(tf.nn.sigmoid(tf.matmul(self.weights['item_embedding'], self.w)))
        """
        *********************************************************
        Inference for the testing phase.
        """
        self.constant_e = self.weights['constant_embedding']
        self.batch_ratings = tf.matmul(self.u_g_embeddings, self.pos_i_g_embeddings, transpose_a=False, transpose_b=True)
        self.batch_ratings_constant = tf.matmul(self.constant_e, self.pos_i_g_embeddings, transpose_a=False, transpose_b=True)
        self.batch_ratings_causal_c = self.batch_ratings - self.batch_ratings_constant
        """
        *********************************************************
        Generate Predictions & Optimize via BPR loss.
        """
        self.mf_loss, self.emb_loss, self.reg_loss = self.create_bpr_loss(self.u_g_embeddings,
                                                                          self.pos_i_g_embeddings,
                                                                          self.neg_i_g_embeddings)
        self.loss = self.mf_loss + self.emb_loss

        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)



        self.mf_loss_bce, self.emb_loss_bce, self.reg_loss_bce = self.create_bce_loss(self.u_g_embeddings,
                                                                          self.pos_i_g_embeddings,
                                                                          self.neg_i_g_embeddings)
        self.loss_bce = self.mf_loss_bce + self.emb_loss_bce
        self.opt_bce = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_bce)



        self.mf_loss_two_bce1, self.emb_loss_two_bce1, self.reg_loss_two_bce1 = self.create_bce_loss_two_brach1(self.u_g_embeddings,
                                                                          self.pos_i_g_embeddings,
                                                                          self.neg_i_g_embeddings)
        self.loss_two_bce1 = self.mf_loss_two_bce1 + self.emb_loss_two_bce1
        self.opt_two_bce1 = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_two_bce1)


        self.mf_loss_two_bce_both, self.emb_loss_two_bce_both, self.reg_loss_two_bce_both = self.create_bce_loss_two_brach_both(self.u_g_embeddings,
                                                                          self.pos_i_g_embeddings,
                                                                          self.neg_i_g_embeddings)
        self.loss_two_bce_both = self.mf_loss_two_bce_both + self.emb_loss_two_bce_both
        self.opt_two_bce_both = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_two_bce_both)



        self.mf_loss_two_bce2, self.emb_loss_two_bce2, self.reg_loss_two_bce2 = self.create_bce_loss_two_brach2(self.u_g_embeddings,
                                                                          self.pos_i_g_embeddings,
                                                                          self.neg_i_g_embeddings)
        self.loss_two_bce2 = self.mf_loss_two_bce2 + self.emb_loss_two_bce2
        self.opt_two_bce2 = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_two_bce2)
    
    
    def create_model_str(self):
        log_dir = '/' + self.alg_type+'/layers_'+str(self.n_layers)+'/dim_'+str(self.emb_dim)
        log_dir+='/'+args.dataset+'/lr_' + str(self.lr) + '/reg_' + str(self.decay)
        return log_dir

    def update_c(self, sess, c):
        sess.run(tf.assign(self.constant_e, c*tf.ones([1, self.emb_dim])))

    def _init_weights(self):
        all_weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()
        all_weights["constant_embedding"] = tf.Variable(tf.ones([1, self.emb_dim]), name='constant_embedding')
        if self.pretrain_data is None:
            all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embedding')
            all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name='item_embedding')
            print('using xavier initialization')
        else:
            all_weights['user_embedding'] = tf.Variable(initial_value=self.pretrain_data['user_embed'], trainable=True,
                                                        name='user_embedding', dtype=tf.float32)
            all_weights['item_embedding'] = tf.Variable(initial_value=self.pretrain_data['item_embed'], trainable=True,
                                                        name='item_embedding', dtype=tf.float32)
            print('using pretrained initialization')

        self.w = tf.Variable(initializer([self.emb_dim, 1]), name = 'item_branch')
        self.w_user = tf.Variable(initializer([self.emb_dim, 1]), name = 'user_branch')
            
        self.weight_size_list = [self.emb_dim] + self.weight_size
        
        for k in range(self.n_layers):
            all_weights['W_gc_%d' %k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_gc_%d' % k)
            all_weights['b_gc_%d' %k] = tf.Variable(
                initializer([1, self.weight_size_list[k+1]]), name='b_gc_%d' % k)

            all_weights['W_bi_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_bi_%d' % k)
            all_weights['b_bi_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_bi_%d' % k)

            all_weights['W_mlp_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_mlp_%d' % k)
            all_weights['b_mlp_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k+1]]), name='b_mlp_%d' % k)

        return all_weights
    def _split_A_hat(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _split_A_hat_node_dropout(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            n_nonzero_temp = X[start:end].count_nonzero()
            A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))

        return A_fold_hat

    def _create_lightgcn_embed(self):
        if self.node_dropout_flag:
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj)
        
        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)
        all_embeddings = [ego_embeddings]
        
        for k in range(0, self.n_layers):

            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            side_embeddings = tf.concat(temp_embed, 0)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings=tf.stack(all_embeddings,1)
        all_embeddings=tf.reduce_mean(all_embeddings,axis=1,keepdims=False)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings
    
    def _create_ngcf_embed(self):
        if self.node_dropout_flag:
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj)

        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):

            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            side_embeddings = tf.concat(temp_embed, 0)
            sum_embeddings = tf.nn.leaky_relu(tf.matmul(side_embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])



            # bi messages of neighbors.
            bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)
            # transformed bi messages of neighbors.
            bi_embeddings = tf.nn.leaky_relu(tf.matmul(bi_embeddings, self.weights['W_bi_%d' % k]) + self.weights['b_bi_%d' % k])
            # non-linear activation.
            ego_embeddings = sum_embeddings + bi_embeddings

            # message dropout.
            ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - self.mess_dropout[k])

            # normalize the distribution of embeddings.
            norm_embeddings = tf.nn.l2_normalize(ego_embeddings, axis=1)

            all_embeddings += [norm_embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings
    
    
    def _create_gcn_embed(self):
        A_fold_hat = self._split_A_hat(self.norm_adj)
        embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)


        all_embeddings = [embeddings]

        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings))

            embeddings = tf.concat(temp_embed, 0)
            embeddings = tf.nn.leaky_relu(tf.matmul(embeddings, self.weights['W_gc_%d' %k]) + self.weights['b_gc_%d' %k])
            embeddings = tf.nn.dropout(embeddings, 1 - self.mess_dropout[k])

            all_embeddings += [embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings
    
    def _create_gcmc_embed(self):
        A_fold_hat = self._split_A_hat(self.norm_adj)

        embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings = []

        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings))
            embeddings = tf.concat(temp_embed, 0)
            # convolutional layer.
            embeddings = tf.nn.leaky_relu(tf.matmul(embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])
            # dense layer.
            mlp_embeddings = tf.matmul(embeddings, self.weights['W_mlp_%d' %k]) + self.weights['b_mlp_%d' %k]
            mlp_embeddings = tf.nn.dropout(mlp_embeddings, 1 - self.mess_dropout[k])

            all_embeddings += [mlp_embeddings]
        all_embeddings = tf.concat(all_embeddings, 1)

        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.nn.sigmoid(tf.reduce_sum(tf.multiply(users, pos_items), axis=1))   #users, pos_items, neg_items have the same shape
        neg_scores = tf.nn.sigmoid(tf.reduce_sum(tf.multiply(users, neg_items), axis=1))
        
        regularizer = tf.nn.l2_loss(self.u_g_embeddings_pre) + tf.nn.l2_loss(
                self.pos_i_g_embeddings_pre) + tf.nn.l2_loss(self.neg_i_g_embeddings_pre)
        regularizer = regularizer / self.batch_size
        
        mf_loss = tf.negative(tf.reduce_mean(tf.log(1e-9+tf.nn.sigmoid(pos_scores - neg_scores))))
        # mf_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))
        
        emb_loss = self.decay * regularizer

        reg_loss = tf.constant(0.0, tf.float32, [1])

        return mf_loss, emb_loss, reg_loss

    def create_bce_loss(self, users, pos_items, neg_items):
        pos_scores = tf.nn.sigmoid(tf.reduce_sum(tf.multiply(users, pos_items), axis=1))   #users, pos_items, neg_items have the same shape
        neg_scores = tf.nn.sigmoid(tf.reduce_sum(tf.multiply(users, neg_items), axis=1))

        regularizer = tf.nn.l2_loss(self.u_g_embeddings_pre) + tf.nn.l2_loss(
                self.pos_i_g_embeddings_pre) + tf.nn.l2_loss(self.neg_i_g_embeddings_pre)
        regularizer = regularizer/self.batch_size

        mf_loss = tf.reduce_mean(tf.negative(tf.log(pos_scores+1e-9))+tf.negative(tf.log(1-neg_scores+1e-9)))
        
        emb_loss = self.decay * regularizer

        reg_loss = tf.constant(0.0, tf.float32, [1])
        
        return mf_loss, emb_loss, reg_loss


    def create_bce_loss_two_brach1(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)   #users, pos_items, neg_items have the same shape
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        # item score
        # pos_items_stop = tf.stop_gradient(pos_items)
        # neg_items_stop = tf.stop_gradient(neg_items)
        pos_items_stop = pos_items
        neg_items_stop = neg_items
        self.pos_item_scores = tf.matmul(pos_items_stop, self.w)
        self.neg_item_scores = tf.matmul(neg_items_stop, self.w)
        self.rubi_ratings1 = (self.batch_ratings-self.rubi_c)*tf.squeeze(tf.nn.sigmoid(self.pos_item_scores))
        self.direct_minus_ratings1 = self.batch_ratings-self.rubi_c*tf.squeeze(tf.nn.sigmoid(self.pos_item_scores))
        # first branch
        # fusion
        pos_scores = pos_scores*tf.nn.sigmoid(self.pos_item_scores)
        neg_scores = neg_scores*tf.nn.sigmoid(self.neg_item_scores)
        self.mf_loss_ori = tf.reduce_mean(tf.negative(tf.log(tf.nn.sigmoid(pos_scores)+1e-10))+tf.negative(tf.log(1-tf.nn.sigmoid(neg_scores)+1e-10)))
        # second branch
        self.mf_loss_item = tf.reduce_mean(tf.negative(tf.log(tf.nn.sigmoid(self.pos_item_scores)+1e-10))+tf.negative(tf.log(1-tf.nn.sigmoid(self.neg_item_scores)+1e-10)))
        # unify
        mf_loss = self.mf_loss_ori + self.alpha*self.mf_loss_item
        # regular
        regularizer = tf.nn.l2_loss(self.u_g_embeddings_pre) + tf.nn.l2_loss(
                self.pos_i_g_embeddings_pre) + tf.nn.l2_loss(self.neg_i_g_embeddings_pre)
        regularizer = regularizer/self.batch_size
        emb_loss = self.decay * regularizer

        reg_loss = tf.constant(0.0, tf.float32, [1])

        return mf_loss, emb_loss, reg_loss

    def create_bce_loss_two_brach2(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)   #users, pos_items, neg_items have the same shape
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        # item score
        # pos_items_stop = tf.stop_gradient(self.pos_i_g_embeddings_pre)
        # neg_items_stop = tf.stop_gradient(self.neg_i_g_embeddings_pre)
        pos_items_stop = self.pos_i_g_embeddings_pre
        neg_items_stop = self.neg_i_g_embeddings_pre
        self.pos_item_scores = tf.matmul(pos_items_stop, self.w)
        self.neg_item_scores = tf.matmul(neg_items_stop, self.w)
        self.rubi_ratings2 = (self.batch_ratings-self.rubi_c)*tf.squeeze(tf.nn.sigmoid(self.pos_item_scores))
        self.direct_minus_ratings2 = self.batch_ratings-self.rubi_c*tf.squeeze(tf.nn.sigmoid(self.pos_item_scores))
        # first branch
        # fusion
        pos_scores = pos_scores*tf.nn.sigmoid(self.pos_item_scores)
        neg_scores = neg_scores*tf.nn.sigmoid(self.neg_item_scores)
        self.mf_loss_ori = tf.reduce_mean(tf.negative(tf.log(tf.nn.sigmoid(pos_scores)+1e-10))+tf.negative(tf.log(1-tf.nn.sigmoid(neg_scores)+1e-10)))
        # second branch
        self.mf_loss_item = tf.reduce_mean(tf.negative(tf.log(tf.nn.sigmoid(self.pos_item_scores)+1e-10))+tf.negative(tf.log(1-tf.nn.sigmoid(self.neg_item_scores)+1e-10)))
        # unify
        mf_loss = self.mf_loss_ori + self.alpha*self.mf_loss_item
        # regular
        regularizer = tf.nn.l2_loss(self.u_g_embeddings_pre) + tf.nn.l2_loss(
                self.pos_i_g_embeddings_pre) + tf.nn.l2_loss(self.neg_i_g_embeddings_pre)
        regularizer = regularizer/self.batch_size
        emb_loss = self.decay * regularizer

        reg_loss = tf.constant(0.0, tf.float32, [1])

        return mf_loss, emb_loss, reg_loss


    def create_bce_loss_two_brach_both(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)   #users, pos_items, neg_items have the same shape
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        # item score
        # pos_items_stop = tf.stop_gradient(pos_items)
        # neg_items_stop = tf.stop_gradient(neg_items)
        pos_items_stop = pos_items
        neg_items_stop = neg_items
        users_stop = users
        self.pos_item_scores = tf.matmul(pos_items_stop, self.w)
        self.neg_item_scores = tf.matmul(neg_items_stop, self.w)
        self.user_scores = tf.matmul(users_stop, self.w_user)
        # self.rubi_ratings_both = (self.batch_ratings-self.rubi_c)*(tf.transpose(tf.nn.sigmoid(self.pos_item_scores))+tf.nn.sigmoid(self.user_scores))
        # self.direct_minus_ratings_both = self.batch_ratings-self.rubi_c*(tf.transpose(tf.nn.sigmoid(self.pos_item_scores))+tf.nn.sigmoid(self.user_scores))
        self.rubi_ratings_both = (self.batch_ratings-self.rubi_c)*tf.transpose(tf.nn.sigmoid(self.pos_item_scores))*tf.nn.sigmoid(self.user_scores)
        self.direct_minus_ratings_both = self.batch_ratings-self.rubi_c*tf.transpose(tf.nn.sigmoid(self.pos_item_scores))*tf.nn.sigmoid(self.user_scores)
        # first branch
        # fusion
        pos_scores = pos_scores*tf.nn.sigmoid(self.pos_item_scores)*tf.nn.sigmoid(self.user_scores)
        neg_scores = neg_scores*tf.nn.sigmoid(self.neg_item_scores)*tf.nn.sigmoid(self.user_scores)
        # pos_scores = pos_scores*(tf.nn.sigmoid(self.pos_item_scores)+tf.nn.sigmoid(self.user_scores))
        # neg_scores = neg_scores*(tf.nn.sigmoid(self.pos_item_scores)+tf.nn.sigmoid(self.user_scores))
        self.mf_loss_ori = tf.reduce_mean(tf.negative(tf.log(tf.nn.sigmoid(pos_scores)+1e-10))+tf.negative(tf.log(1-tf.nn.sigmoid(neg_scores)+1e-10)))
        # second branch
        self.mf_loss_item = tf.reduce_mean(tf.negative(tf.log(tf.nn.sigmoid(self.pos_item_scores)+1e-10))+tf.negative(tf.log(1-tf.nn.sigmoid(self.neg_item_scores)+1e-10)))
        # third branch
        self.mf_loss_user = tf.reduce_mean(tf.negative(tf.log(tf.nn.sigmoid(self.user_scores)+1e-10))+tf.negative(tf.log(1-tf.nn.sigmoid(self.user_scores)+1e-10)))
        # unify
        mf_loss = self.mf_loss_ori + self.alpha*self.mf_loss_item + self.beta*self.mf_loss_user
        # regular
        regularizer = tf.nn.l2_loss(self.u_g_embeddings_pre) + tf.nn.l2_loss(
                self.pos_i_g_embeddings_pre) + tf.nn.l2_loss(self.neg_i_g_embeddings_pre)
        regularizer = regularizer/self.batch_size
        emb_loss = self.decay * regularizer

        reg_loss = tf.constant(0.0, tf.float32, [1])

        return mf_loss, emb_loss, reg_loss



    
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)
        
    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        """
        Dropout for sparse tensors.
        """
        noise_shape = [n_nonzero_elems]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse_retain(X, dropout_mask)

        return pre_out * tf.div(1., keep_prob)
    
    def update_c(self, sess, c):
        sess.run(tf.assign(self.rubi_c, c*tf.ones([1])))

def load_pretrained_data():
    pretrain_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, 'embedding')
    try:
        pretrain_data = np.load(pretrain_path)
        print('load the pretrained embeddings.')
    except Exception:
        pretrain_data = None
    return pretrain_data

# parallelized sampling on CPU 
class sample_thread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self):
        with tf.device(cpus[0]):
            self.data = data_generator.sample()

class sample_thread_test(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self):
        with tf.device(cpus[0]):
            self.data = data_generator.sample_test()
            
# training on GPU
class train_thread(threading.Thread):
    def __init__(self,model, sess, sample, args):
        threading.Thread.__init__(self)
        self.model = model
        self.sess = sess
        self.sample = sample
    def run(self):
        sess_list = []
        if args.loss == 'bpr':
            sess_list = [self.model.opt, self.model.loss, self.model.mf_loss, self.model.emb_loss, self.model.reg_loss]
        elif args.loss == 'bce':
            sess_list = [self.model.opt_bce, self.model.loss_bce, self.model.mf_loss_bce, self.model.emb_loss_bce, self.model.reg_loss_bce]
        elif args.loss == 'bce1':
            sess_list = [self.model.opt_two_bce1, self.model.loss_two_bce1, self.model.mf_loss_two_bce1, self.model.emb_loss_two_bce1, self.model.reg_loss_two_bce1]
        elif args.loss == 'bce2':
            sess_list = [self.model.opt_two_bce2, self.model.loss_two_bce2, self.model.mf_loss_two_bce2, self.model.emb_loss_two_bce2, self.model.reg_loss_two_bce2]
        elif args.loss == 'bceboth':
            sess_list = [self.model.opt_two_bce_both, self.model.loss_two_bce_both, self.model.mf_loss_two_bce_both, self.model.emb_loss_two_bce_both, self.model.reg_loss_two_bce_both]
        if len(gpus):
            with tf.device(gpus[-1]):
                users, pos_items, neg_items = self.sample.data
                self.data = sess.run(sess_list,
                                       feed_dict={model.users: users, model.pos_items: pos_items,
                                                  model.node_dropout: eval(args.node_dropout),
                                                  model.mess_dropout: eval(args.mess_dropout),
                                                  model.neg_items: neg_items})
        else:
            users, pos_items, neg_items = self.sample.data
            self.data = sess.run(sess_list,
                                       feed_dict={model.users: users, model.pos_items: pos_items,
                                                  model.node_dropout: eval(args.node_dropout),
                                                  model.mess_dropout: eval(args.mess_dropout),
                                                  model.neg_items: neg_items})
class train_thread_test(threading.Thread):
    def __init__(self, model, sess, sample, args):
        threading.Thread.__init__(self)
        self.model = model
        self.sess = sess
        self.sample = sample
    def run(self):
        sess_list = []
        if args.loss == 'bpr':
            sess_list = [self.model.loss, self.model.mf_loss, self.model.emb_loss]
        elif args.loss == 'bce':
            sess_list = [self.model.loss_bce, self.model.mf_loss_bce, self.model.emb_loss_bce]
        elif args.loss == 'bce1':
            sess_list = [self.model.loss_two_bce1, self.model.mf_loss_two_bce1, self.model.emb_loss_two_bce1]
        elif args.loss == 'bce2':
            sess_list = [self.model.loss_two_bce2, self.model.mf_loss_two_bce2, self.model.emb_loss_two_bce2]
        elif args.loss == 'bceboth':
            sess_list = [self.model.loss_two_bce_both, self.model.mf_loss_two_bce_both, self.model.emb_loss_two_bce_both]
        if len(gpus):
            with tf.device(gpus[-1]):
                users, pos_items, neg_items = self.sample.data
                self.data = sess.run(sess_list,
                                     feed_dict={model.users: users, model.pos_items: pos_items,
                                                model.neg_items: neg_items,
                                                model.node_dropout: eval(args.node_dropout),
                                                model.mess_dropout: eval(args.mess_dropout)})
        else:
            users, pos_items, neg_items = self.sample.data
            self.data = sess.run(sess_list,
                                     feed_dict={model.users: users, model.pos_items: pos_items,
                                                model.neg_items: neg_items,
                                                model.node_dropout: eval(args.node_dropout),
                                                model.mess_dropout: eval(args.mess_dropout)})            

if __name__ == '__main__':
    f0 = time()
    seed = 12345
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    logging.basicConfig(filename="LightGCN_{}_{}_{}_{}".format(args.dataset, args.loss, args.test, args.alpha))
    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    """
    *********************************************************
    Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
    """
    plain_adj, norm_adj, mean_adj, pre_adj = data_generator.get_adj_mat()
    if args.adj_type == 'plain':
        config['norm_adj'] = plain_adj
        print('use the plain adjacency matrix')
    elif args.adj_type == 'norm':
        config['norm_adj'] = norm_adj
        print('use the normalized adjacency matrix')
    elif args.adj_type == 'gcmc':
        config['norm_adj'] = mean_adj
        print('use the gcmc adjacency matrix')
    elif args.adj_type=='pre':
        config['norm_adj']=pre_adj
        print('use the pre adjcency matrix')
    else:
        config['norm_adj'] = mean_adj + sp.eye(mean_adj.shape[0])
        print('use the mean adjacency matrix')
    t0 = time()
    if args.pretrain == -1:
        pretrain_data = load_pretrained_data()
    else:
        pretrain_data = None
    model = LightGCN(data_config=config, pretrain_data=pretrain_data)
    
    """
    *********************************************************
    Save the model parameters.
    """
    saver = tf.train.Saver()

    if args.save_flag == 1:
        layer = '-'.join([str(l) for l in eval(args.layer_size)])
        weights_save_path = '%sweights/%s/%s/%s/l%s_r%s' % (args.weights_path, args.dataset, model.model_type, layer,
                                                            str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))
        ensureDir(weights_save_path)
        save_saver = tf.train.Saver(max_to_keep=20000)

    """
    *********************************************************
    Reload the pretrained model parameters.
    """
    
    if args.pretrain == 1:
        # MACR
        model_file = "Your Path"
        best_c = 45 # value of c
        users_to_test = list(data_generator.test_set.keys())
        saver.restore(sess, model_file)
        for c in [0, best_c]:
            model.update_c(sess, c)
            ret = test(sess, model, users_to_test, method="rubiboth")
            perf_str = 'c:{}: recall={}, hit={}, ndcg={}'.format(c, str(ret["recall"]),
                                             str(ret['hr']), str(ret['ndcg']))
            print(perf_str)
        exit()

    elif args.pretrain == 0:
        sess.run(tf.global_variables_initializer())
        cur_best_pre_0 = 0.
        print('without pretraining.')

    """
    *********************************************************
    Train.
    """
    tensorboard_model_path = 'tensorboard/'
    if not os.path.exists(tensorboard_model_path):
        os.makedirs(tensorboard_model_path)
    run_time = 1
    while (True):
        if os.path.exists(tensorboard_model_path + model.log_dir +'/run_' + str(run_time)):
            run_time += 1
        else:
            break
    train_writer = tf.summary.FileWriter(tensorboard_model_path +model.log_dir+ '/run_' + str(run_time), sess.graph)
    
    
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    config = dict()
    config["best_hr"], config["best_ndcg"], config['best_recall'], config['best_pre'], config["best_epoch"] = 0, 0, 0, 0, 0
    config['best_c_hr'], config['best_c_epoch'] = 0, 0
    stopping_step = 0
    should_stop = False
    
    best_epoch=0
    best_hr_norm = 0
    best_str = ''
    if args.only_test == 0 and args.pretrain == 0:
        for epoch in range(1, args.epoch + 1):
            t1 = time()
            loss, mf_loss, emb_loss, reg_loss = 0., 0., 0., 0.
            n_batch = data_generator.n_train // args.batch_size + 1
            loss_test,mf_loss_test,emb_loss_test,reg_loss_test=0.,0.,0.,0.
            '''
            *********************************************************
            parallelized train sampling
            '''
            sample_last = sample_thread()
            sample_last.start()
            sample_last.join()
            for idx in range(n_batch):
                train_cur = train_thread(model, sess, sample_last, args)
                sample_next = sample_thread()
                
                train_cur.start()
                sample_next.start()
                
                sample_next.join()
                train_cur.join()
                
                users, pos_items, neg_items = sample_last.data
                _, batch_loss, batch_mf_loss, batch_emb_loss, batch_reg_loss = train_cur.data
                sample_last = sample_next
            
                loss += batch_loss/n_batch
                mf_loss += batch_mf_loss/n_batch
                emb_loss += batch_emb_loss/n_batch
                
            if np.isnan(loss) == True:
                print('ERROR: loss is nan.')
                sys.exit()
            if (epoch % args.log_interval) != 0:
                if args.verbose > 0 and epoch % args.verbose == 0:
                    perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                        epoch, time() - t1, loss, mf_loss, emb_loss)
                    print(perf_str)
                    logging.info(perf_str)
                continue
            
            '''
            *********************************************************
            parallelized test sampling
            '''
            # test loss
            sample_last= sample_thread_test()
            sample_last.start()
            sample_last.join()
            loss_test,mf_loss_test,emb_loss_test,reg_loss_test=0.,0.,0.,0.
            for idx in range(n_batch):
                train_cur = train_thread_test(model, sess, sample_last, args)
                sample_next = sample_thread_test()
                
                train_cur.start()
                sample_next.start()
                
                sample_next.join()
                train_cur.join()
                
                users, pos_items, neg_items = sample_last.data
                batch_loss_test, batch_mf_loss_test, batch_emb_loss_test = train_cur.data
                sample_last = sample_next
                
                loss_test += batch_loss_test / n_batch
                mf_loss_test += batch_mf_loss_test / n_batch
                emb_loss_test += batch_emb_loss_test / n_batch
                
            t2 = time()
            users_to_test = list(data_generator.test_set.keys())


            perf_str = ''
            if args.test == 'normal':
                ret = test(sess, model, users_to_test, drop_flag=True)                                                                                 
                t3 = time()
                
                loss_loger.append(loss)
                rec_loger.append(ret['recall'])
                pre_loger.append(ret['hr'])
                ndcg_loger.append(ret['ndcg'])

                if args.verbose > 0:
                    perf_str = 'Epoch %d [%.1fs + %.1fs]: test==[%.5f=%.5f + %.5f + %.5f], recall=[%s], ' \
                            'hr=[%s], ndcg=[%s]\n' % \
                            (epoch, t2 - t1, t3 - t2, loss_test, mf_loss_test, emb_loss_test, reg_loss_test, 
                                ', '.join(['%.5f' % r for r in ret['recall']]),
                                ', '.join(['%.5f' % r for r in ret['hr']]),
                                ', '.join(['%.5f' % r for r in ret['ndcg']]))
                    print(perf_str, end='')
                    logging.info(perf_str)
                if ret['hr'][0] > best_hr_norm:
                    best_hr_norm = ret['hr'][0]
                    best_epoch = epoch
                    best_str = perf_str
            elif args.test=="rubi1" or args.test=='rubi2' or args.test=='rubiboth':
                print('Epoch %d'%(epoch))
                best_c = 0
                best_hr = 0
                for c in np.linspace(args.start, args.end, args.step):
                    model.update_c(sess, c)
                    ret = test(sess, model, users_to_test, method=args.test)
                    t3 = time()
                    loss_loger.append(loss)
                    rec_loger.append(ret['recall'][0])
                    ndcg_loger.append(ret['ndcg'][0])
                    hit_loger.append(ret['hr'][0])

                    if ret['hr'][0] > best_hr:
                        best_hr = ret['hr'][0]
                        best_c = c

                    if args.verbose > 0:
                        perf_str += 'c:%.2f recall=[%.5f, %.5f], ' \
                                    'hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]\n' % \
                                    (c, ret['recall'][0], ret['recall'][-1],
                                        ret['hr'][0], ret['hr'][-1],
                                        ret['ndcg'][0], ret['ndcg'][-1])
                
                flg = False
                    
                ret['hr'][0] = best_hr
                print(perf_str, end='')
                logging.info(perf_str)
                if flg:
                    best_str = perf_str



            
                
            cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['hr'][0], cur_best_pre_0,
                                                                        stopping_step, expected_order='acc', flag_step=10)

            # *********************************************************
            # save the user & item embeddings for pretraining.
            if ret['hr'][0] == cur_best_pre_0:
                best_epoch = epoch
            if args.save_flag == 1:
                save_saver.save(sess, weights_save_path + '/weights_{}'.format(args.saveID), global_step=epoch)
                print('save the weights in path: ', weights_save_path)
            
            # *********************************************************
            # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
            if should_stop == True and args.early_stop == 1:
                if args.test != 'normal':
                    with open(weights_save_path + '/best_epoch_{}.txt'.format(args.saveID),'w') as f:
                        f.write(str(config['best_c_epoch']))
                else:
                    with open(weights_save_path + '/best_epoch_{}.txt'.format(args.saveID),'w') as f:
                        f.write(str(best_epoch))
                break

        