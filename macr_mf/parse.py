import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run pop_bias.")
    parser.add_argument('--data_path', nargs='?', default='./data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='movielens_ml_1m',
                        help='Choose a dataset')
    parser.add_argument('--source', nargs='?', default='normal',
    help='normal | dice')
    parser.add_argument('--train', nargs='?', default='normalbce',
    help='normalbce | rubibceboth')
    parser.add_argument('--test', nargs='?', default='normal',
    help='normal | rubi')
    parser.add_argument('--valid_set', nargs='?', default='test',
    help='test | valid')
    parser.add_argument('--alpha', type=float, default=1e-3,
                        help='alpha')
    parser.add_argument('--beta', type=float, default=1e-3,
                        help='beta')
    parser.add_argument('--early_stop', type=int, default=1,
                        help='alpha')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=400,
                        help='Number of epoch.')
    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')
    parser.add_argument('--Ks', nargs='?', default='[20]',
                        help='Evaluate on Ks optimal items.')
    parser.add_argument('--epochs', nargs='?', default='[]',
                        help='Test c on these epochs.')
    parser.add_argument('--regs', type=float, default=1e-5,
                        help='Regularizations.')
    parser.add_argument('--c', type=float, default=10.0,
                        help='Constant c.')
    parser.add_argument('--train_c', type=str, default="val",
                        help='val | test')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate.')
    parser.add_argument('--wd', type=float, default=1e-5,
                        help='Weight decay of optimizer.')
    parser.add_argument('--model', nargs='?', default='mf',
                        help='Specify model type, choose from {mf, CausalE}')
    parser.add_argument('--skew', type=int, default=0,
                        help='Use not skewed dataset.')
    # parser.add_argument('--model_type', nargs='?', default='o',
    #                     help='Specify model type, choose from {o, c, ic, rc, irc}')
    parser.add_argument('--devide_ratio', type=float, default=0.8,
                        help='Train/Test.')
    parser.add_argument('--save_flag', type=int, default=1,
                        help='0: Disable model saver, 1: Activate model saver')
    parser.add_argument('--cuda', type=str, default='1',
                        help='Avaiable GPU ID')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='0: no pretrain, 1: load pretrain model')
    parser.add_argument('--check_c', type=int, default=1,
                        help='0: no checking, 1: check a range of cs')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='log\'s interval epoch while training')
    parser.add_argument('--pop_wd', type=float, default=0.,
                        help='weight decay of popularity')
    parser.add_argument('--base', type=float, default=-1.,
                        help='check range base.')
    parser.add_argument('--cf_pen', type=float, default=1.0,
                        help='Imbalance loss.')
    parser.add_argument('--saveID', nargs='?', default='',
                        help='Specify model save path.')
    parser.add_argument('--user_min', type=int, default=1,
                        help='user_min.')
    parser.add_argument('--user_max', type=int, default=1000,
                        help='user max per cls.')
    parser.add_argument('--data_type', nargs='?', default='ori',
                        help='load imbalanced data or not.')
    parser.add_argument('--imb_type', nargs='?', default='exp',
                        help='imbalance type.')
    parser.add_argument('--top_ratio', type=float, default=0.1,
                        help='imbalance top ratio.')
    parser.add_argument('--lam', type=float, default=1.,
                        help='lambda.')
    parser.add_argument('--check_epoch', nargs='?', default='all',
                        help='check all epochs or select some or search in range.')  
    parser.add_argument('--start', type=float, default=-1.,
                        help='check c start.')
    parser.add_argument('--end', type=float, default=1.,
                        help='check c end.')
    parser.add_argument('--step', type=int, default=20,
                        help='check c step.')      
    parser.add_argument('--out', type=int, default=0)                      
    return parser.parse_args()
