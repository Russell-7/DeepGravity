import numpy as np
import torch
from scipy.stats import entropy

def CPC(a, b):
    if ((a < 0).sum() + (b < 0).sum()) > 0:
        raise("OD flow should not be less than zero.")
    if type(a) == type(np.array([1, 1])):
        a = np.round(a)
        b = np.round(b)
        min = np.minimum(a, b)
        return 2 * min.sum() / ( a.sum() + b.sum())
    else:
        a = a.round()
        b = b.round()
        min = torch.minimum(a, b)
        return 2 * min.sum() / ( a.sum() + b.sum())
def CPC_nonzero(a, b):
    if type(a) == type(np.array([1, 1])):
        idx = b.nonzero()
        a, b = a[idx], b[idx]
    else:
        idx = b.nonzero()
        idx = (idx[:, 0], idx[:, 1])
        a, b = a[idx], b[idx]
    return CPC(a, b)

def JSD_ODflow(a, b):
    if type(a) == type(np.array([1, 1])):
        a = torch.FloatTensor(a)
    if type(b) == type(np.array([1, 1])):
        b = torch.FloatTensor(b)
    a, b = a.cpu().numpy().reshape([-1]), b.cpu().numpy().reshape([-1])
    sections, b_dist = values_to_bucket(b)
    # print(sections)
    # print(b_dist)
    a_dist = []
    for i in range(len(sections) - 1):
        low, high = sections[i], sections[i+1]
        frequency = np.sum( (a >= low) & (a < high) )
        a_dist.append(frequency)
    
    a_dist = np.array(a_dist) / np.array(a_dist).sum()
    b_dist = np.array(b_dist) / np.array(b_dist).sum()
    # print(a_dist)
    # print(b_dist)

    return JS_divergence(a_dist, b_dist)

def JS_divergence(p, q):
    M = (p + q) / 2
    return 0.5 * entropy(p, M, base=2) + 0.5 * entropy(q, M, base=2)

def values_to_bucket(values):

    # 2的指数分桶
    max_ = values.max()
    # print(max_)
    i = 0
    leftright = []
    nums = []
    while True:
        if i == 0:
            left = 0
            right = 1
            leftright.append(left)
            leftright.append(right)
            i += 1
        else:
            left = i
            right = i * 2
            leftright.append(right)
            i = i *2
        nums.append(((values >= left) & (values <right)).sum())

        if right > max_:
            break
    return leftright, nums

