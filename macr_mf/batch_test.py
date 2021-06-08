from parse import parse_args
from load_data import Data
import multiprocessing
import heapq

args = parse_args()
data = Data(args)
sorted_id, belong, rate, usersorted_id, userbelong, userrate = data.plot_pics()
Ks = eval(args.Ks)
BATCH_SIZE = args.batch_size
ITEM_NUM = data.n_items
USER_NUM = data.n_users

points = [10, 50, 100, 200, 500]