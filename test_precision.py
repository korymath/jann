import time
import random
from annoy import AnnoyIndex


num_trees = 100
num_lines = 2048
t = AnnoyIndex(512)
t.load('data/CMDC/all_lines_{}.txt.ann'.format(num_lines))

n = t.get_n_items()
print('{} items in the tree.'.format(n))

limits = [10, 100, 1000, 10000, 100000]
k = 10
prec_sum = {}
prec_n = 1000
time_sum = {}

for i in range(prec_n):
    j = random.randrange(0, n)

    closest = set(t.get_nns_by_item(j, k, search_k=n))
    for limit in limits:
        t0 = time.time()
        toplist = t.get_nns_by_item(j, k, search_k=limit)
        T = time.time() - t0

        found = len(closest.intersection(toplist))
        hitrate = 1.0 * found / k
        prec_sum[limit] = prec_sum.get(limit, 0.0) + hitrate
        time_sum[limit] = time_sum.get(limit, 0.0) + T

for limit in limits:
    print('limit: %-9d precision: %6.2f%% avg time: %.6fs'
          % (limit, 100.0 * prec_sum[limit] / (i + 1),
             time_sum[limit] / (i + 1)))
