import pickle
import numpy as np
import os

def pkl_save(obj, dir, file_name):
    file_path = os.path.join(dir, f'{file_name}.pkl')
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)
    file_size = os.path.getsize(file_path)
    print(f'{file_name}.pkl // FILE SIZE: {file_size / 1024**2 :.0f} MB')

def pkl_load(dir, file_name):
    file_path = os.path.join(dir, f'{file_name}.pkl')
    obj = pickle.load(open(file_path, 'rb'))
    file_size = os.path.getsize(file_path)
    print(f'{file_name}.pkl // FILE SIZE: {file_size / 1024**2 :.0f} MB')

    return obj

def get_info(df, columns, dir, file_name):
    info = {c:{} for c in columns}
    for c in columns:
        mean = df[c].mean()
        std = df[c].std()
        info[c]['mean'] = mean
        info[c]['std'] = std
    pkl_save(info, dir, file_name)
    display(info)
    
    return info

def gaussian_kernel(length: int, sigma: int, dtype: type) -> np.ndarray:
    x = np.ogrid[-length : length + 1]
    h = np.exp(-(x**2) / (2 * sigma * sigma))
    h[h < np.finfo(dtype).eps * h.max()] = 0
    
    return h

def gaussian_label(label: np.ndarray, offset: int, sigma: int, dtype: type) -> np.ndarray:
    num_events = label.shape[1]
    for i in range(num_events):
        label[:, i] = np.convolve(label[:, i], gaussian_kernel(offset, sigma, dtype), mode="same")

    return label

def list_split(lst, n_splits):
    n_per_split = len(lst) / n_splits
    split_lst = []
    for i in range(n_splits):
        st = int(i * n_per_split)
        ed = int((i+1) * n_per_split)
        split_lst.append(lst[st:ed])
    
    return split_lst

def train_valid_split(lst, n_splits):
    valid = list_split(lst, n_splits)
    train = []
    for fold, f_vlst in enumerate(valid):
        f_tlst = list(set(lst) - set(f_vlst))
        f_tlst.sort()
        train.append(f_tlst)
        print(fold, ' // ', len(f_tlst) , ' // ', len(f_vlst))

    return train, valid