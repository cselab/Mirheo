#!/usr/bin/env python

def ellipsoids(domain, axes, numberdensity, q):

    maxNum     = [domain[i] / (2*axes[i]) for i in range(3)]
    domain_vol = domain[0] * domain[1] * domain[2]
    nobjs      = domain_vol * numberdensity

    if maxNum[0] * maxNum[1] * maxNum[2] < nobjs:
        return []

    A = (numberdensity * axes[0] * axes[1] * axes[2]) ** (1.0 / 3.0)
    h = [axes[i] / A for i in range(3)]
    n = [int (max([1.0, domain[i] / h[i]])) for i in range(3)]
    h = [domain[i] / n[i] for i in range(3)]    
    
    com_q = []
    for i in range(n[0]):
        for j in range(n[1]):
            for k in range(n[2]):
                com_q.append( [(i+0.5) * h[0],
                               (j+0.5) * h[1],
                               (k+0.5) * h[2],
                               q[0], q[1], q[2], q[3]] )

    return n, com_q


if __name__ == '__main__':
    import numpy as np
    domain     = (32, 32, 102.0)
    axes       = (6.0, 1.0, 1.0)
    numdensity = 6.25e-05
    q = [1, 0, 0, 0]
    (n, com_q) = ellipsoids(domain, axes, numdensity)
    print(n)
    print(np.array(com_q)[:, 0:3])
