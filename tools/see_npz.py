import numpy as np

path = '/work2/08264/baagee/frontera/site-response/eq-data/FKSH17/data_all_freq.npz'
data = [dict(trj_info.item()) for trj_info in np.load(path, allow_pickle=True).values()]
a=1