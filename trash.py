#%%
import numpy as np

def is_in_circ(x, y, cx, cy, r):
    return (x - cx)**2 + (y - cy)**2 <= r**2
        


#%%
xs = np.random.uniform(-6, 7, 10_000_000)
ys = np.random.uniform(-7, 6, 10_000_000)

#%%
c1 = is_in_circ(xs, ys, 0, 0, 6)

c2 = is_in_circ(xs, ys, 5, -5, 2)

#%%
(c1 & c2).mean() * 13 * 13