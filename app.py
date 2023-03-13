import faiss
import os
import sys
import numpy as np

# 从文件中读取向量
"""
fea_v_matrix: np.array形式的特征向量矩阵
fea_s_list: id的顺序列表
key_d: id和位置的对应dict
"""
def load_fea_from_file(fname):
    line_num = 0
    fea_s_list = list()
    fea_v_list = list()
    
    key_d = dict()
    idx = 0
    with open(fname, "r", encoding="utf-8", errors="replace") as f:
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
    fea_v_matrix = np.array(fea_v_list)
    return fea_v_matrix, fea_s_list, key_d
def build_index_fun(path):
    # path : './item_vec.txt'
    fv_mat, fea_s_list, key_d = load_fea_from_file(path)
    print(fv_mat)
    print('-------------------')
    print(fea_s_list)
    print('-------------------')
    print(key_d)

    # cluster_num 是聚类的数量
    M = fv_mat.shape[1] // 4
    cluster_num = int(fv_mat.shape[0]) // 14
    # metric_type
    metric_type = faiss.METRIC_INNER_PRODUCT
    # 创建索引
    print(M, cluster_num, "aaaaaaaaaaaaaaaa")
    index = faiss.index_factory(fv_mat.shape[1], f"IVF{cluster_num},PQ{M}x4fs", metric_type)

    # 正则化矩阵特征向量
    faiss.normalize_L2(fv_mat)

    # 训练索引
    index.train(fv_mat)

    # 索引添加矩阵向量
    index.add(fv_mat)

    # 创建一个精度索引， 它细化了距离计算，并根据这些距离对结果重新排序
    index_refine = faiss.IndexRefineFlat(index, faiss.swig_ptr(fv_mat))
    return index, index_refine, fea_s_list, key_d

# 将faiss的索引index写到文件中
def write_index(path):
    # path = './item_vec.txt'
    index, index_refine, keys, key_d = build_index_fun(path)
    faiss.write_index(index, './vec/index_m.faiss')
    faiss.write_index(index_refine, './vec/index_refine_m.faiss')
    # 商品和序列的键值对也写到文件中
    with open('./vec/index_key.faiss', "w") as fout:
        for ky in keys:
            fout.write(ky)
            fout.write("\n")
    return index, index_refine, keys, key_d

write_index('./item_vec.txt')

def load_index(index_dir):
    # 加载索引文件
    index = faiss.read_index('./vec/index_m.faiss')
    index_refine = faiss.read_index('./vec/index_m.faiss')
    # 设置基础索引
    index_refine.base_index = index
    index_refine.k_factor = 10
    index.nprobe = 100
    # 商品位置键值对
    keys = list()
    key_d = dict()
    with open("./vec/index_key.faiss") as fin:
        idx = 0
        for line in fin:
            line = line.strip("\r\n")
            key_d[line] = idx
            idx += 1
            keys.append(line)
    return index, index_refine, keys, key_d
    
# 向量检索
def search():
    # 输入一个index库里的向量，进行检索
    index, index_refine, keys, key_d = load_index('./item_vec.txt')
    
    # 获取一个商品的id
    fs = '653417384213'
    i = key_d[fs]
    # 根据位置获取
    index_refine.make_direct_map()
    fv = index_refine.reconstruct(1)
    print(fv)
    
   
 
# write_index('./item_vec.txt')
search()
    
