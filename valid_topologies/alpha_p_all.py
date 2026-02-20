import math
import numpy as np
import itertools

def topologies():
    N = 10
    B = 3
    A = N - B
    L = 3

    combinations = itertools.product(range(A + 1), repeat=L)
    al = [np.array(tuple) for tuple in combinations if sum(tuple) == A]

    combinations2 = itertools.product(range(B + 1), repeat=L)
    bl = [np.array(tuple) for tuple in combinations2 if sum(tuple) == B]
    res = 0
    Produit1 = math.factorial(A) * math.factorial(B)
    Produit2 = math.pow(L, A) * math.pow(L, B)
    for a in al:
        for b in bl:
            var1 = 1
            for i in range(len(a)):
                if a[i] + b[i] == 0:
                    var1 *= 0
                else:
                    var1 *= b[i] / (a[i] + b[i])
            var2 = np.product([math.factorial(ai) for ai in a])
            var3 = np.product([math.factorial(bi) for bi in b])
            res += (var1 * Produit1) / (var2 * var3 * Produit2)
    return res
if __name__ == "__main__":
    pr_F = topologies()
    print("All topologies")
    print("PrF", pr_F)