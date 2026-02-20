import math
import numpy as np
import itertools
import cProfile
import functools
from time import time
from tqdm import tqdm

PROFILE = False


@functools.lru_cache(maxsize=2**30)
def memoized_factorial(n):
    return math.factorial(n)


@functools.lru_cache(maxsize=2**30)
def memoized_compute_var3(b):
    return np.product([memoized_factorial(bi) for bi in b])


def run():
    N = 10
    B = 3
    A = N - B
    L = 3
    t0 = time()
    combinations = itertools.product(range(A + 1), repeat=L)
    # make al a generator expression so it's doesn't take up memory
    al = [np.array(tuple_) for tuple_ in combinations if sum(tuple_) == A]
    # al = (np.array(tuple) for tuple in combinations if sum(tuple) == A)
    t1 = time()
    if PROFILE:
        print("al done in %0.1f" % (t1-t0))

    combinations2 = itertools.product(range(B + 1), repeat=L)

    # we can't make bl a generator expression since it's used multiple times
    bl = [np.array(tuple_) for tuple_ in combinations2 if sum(tuple_) == B]
    t2 = time()
    if PROFILE:
        print("bl done in %0.1f" % (t2-t1))

    pr_F = 0
    Produit1 = memoized_factorial(A) * memoized_factorial(B)
    Produit2 = math.pow(L, A) * math.pow(L, B)
    n_inner_loop_cnt = len(al) * len(bl)
    INNERLOOP_SPEED = 170000  # loop per sec
    print("Inner loop will run %d times" % n_inner_loop_cnt)
    print("Simulation is expected to take ~%d seconds" % (
        n_inner_loop_cnt/INNERLOOP_SPEED))

    # for a in al:
    for a in tqdm(al):  # measure the loop speed
        len_a = len(a)
        var2 = np.product([memoized_factorial(ai) for ai in a])

        # precompute the following two variables, so you don't have to do in
        # the inner loop.
        var2_times_Produit2 = var2 * Produit2
        Produit1_div_by_var2_times_Produit2 = Produit1 / var2_times_Produit2

        for b in bl:
            var1 = 1
            for i in range(len_a):
                ab_sum = a[i] + b[i]  # so we don't add the same numbers twice
                if ab_sum == 0:
                    var1 = 0
                    break  # no need to continue
                else:
                    var1 *= b[i] / ab_sum

            # var3 = np.product([factorial(bi) for bi in b])
            var3 = memoized_compute_var3(tuple(b))
            pr_F += (var1 * Produit1_div_by_var2_times_Produit2) / var3
    print("All topologies")
    print("PrF", pr_F)


if PROFILE:
    cProfile.run('run()', sort='cumulative')
    print("=== Cache hits ===")
    print("memoized_compute_var3.cache_info",
          memoized_compute_var3.cache_info())
    print("memoized_factorial.cache_info", memoized_factorial.cache_info())
else:
    run()
