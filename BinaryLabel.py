import os
import numpy as np

data = "/home/group10/deephalo_gnn/Labeled subhalo matrices of haloes"
data2 = "/home/group10/deephalo_gnn/BinaryLabeledData"
files = os.listdir(data)
point_cloud_data = [(np.load(data+"/"+f)) for f in files if f.endswith(".npy")]
point_cloud_data = sorted(point_cloud_data, key=lambda x: x.size, reverse=True)
s = 1
for i in point_cloud_data:
    l = i[:,-1]
    l[l>-1] = 0
    l+=1
    i[:,-1] = l
    np.save(data2+"/"+str(s), i)
    s+=1