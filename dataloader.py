import os

import numpy as np
from tqdm import tqdm


def do_remap(old_array):
    array = np.array(old_array)
    field_num = np.shape(array)[-1] - 2  # last two is time and label
    fields_feature_num = np.zeros(field_num, dtype=np.int32)  # 记录每个特征域最大特征数目
    fields_feature_num[0] = np.max(array[:, 0]) + 1
    for field in range(1, field_num):
        remap_index = np.unique(array[:, field], return_inverse=True)[1]
        fields_feature_num[field] = np.max(remap_index) + 1
        array[:, field] = remap_index + np.sum(fields_feature_num[:field])
    return array, fields_feature_num


def get_sub_sequence(user_item, dataset_name, new_folder='./Data', max_len=20):
    old_folder = 'Data/{}/'.format(dataset_name)
    new_folder = os.path.join(new_folder, dataset_name)
    if not os.path.exists(new_folder):
        os.mkdir(new_folder)
    with  np.load(os.path.join(old_folder, user_item)) as user_item:
        log, begin_len = user_item['log'], user_item['begin_len']
    every_last_index = []
    for (begin_loc, seq_len) in tqdm(begin_len):
        end_loc = begin_loc + seq_len
        every_last_index += list(range(end_loc - max_len, end_loc))
    log_user_last = log[every_last_index]
    log_user_last, fields_feature_num = do_remap(log_user_last)
    print(fields_feature_num)
    log_user_last = np.reshape(log_user_last, (-1, max_len, log_user_last.shape[-1]))
    print(fields_feature_num.dtype)
    np.savez(os.path.join(new_folder, 'user_item.npz'),
             log=log_user_last.astype(np.int32),
             fields=fields_feature_num
             )


if __name__ == '__main__':
    get_sub_sequence('user_item.npz', 'alipay', max_len=30)
