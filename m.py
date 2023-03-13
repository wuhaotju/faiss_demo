import faiss
import numpy as np
import time


import faiss
import numpy as np
import time

faiss.omp_set_num_threads(1)
nb_vectors = 100000
dimension = 8
vectors = np.random.rand(nb_vectors, dimension).astype('float32')

flat_index = faiss.index_factory(vectors.shape[1], "IVF100,PQ8")
flat_index.train(vectors)
flat_index.add(vectors)

N = 10000
start_time = time.perf_counter()

flat_index.make_direct_map()
v = flat_index.reconstruct(10)
print(v)
  
"""
line_num = 0
key_d = dict()
fea_s_list = []
fea_v_list = []
idx=  0 
with open('./item_vec.txt', "r", encoding="utf-8", errors="replace") as f:
        for row_content in f:
            line_num += 1
            
            try:
                raw_content_list = row_content.strip().split()
                fea_s = raw_content_list[0]
                fea_v = np.array(raw_content_list[1:]).astype(np.float32)
            except Exception as e:
                sys.exit("fail to load file vector")
                
            if fea_s not in key_d:
                key_d[fea_s] = idx
                idx += 1
                fea_s_list.append(fea_s)
                fea_v_list.append(fea_v)
fv_mat = np.array(fea_v_list)
    
index = faiss.index_factory(fv_mat.shape[1], "IVF100,PQ8")
fv = index.reconstruct(1)
print(fv)
"""