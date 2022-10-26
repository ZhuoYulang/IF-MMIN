import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
from utils.logger import get_logger, ResultRecorder, LossRecorder
import pickle
import numpy as np
import random
from sklearn.manifold import TSNE
import os
import warnings

warnings.filterwarnings("ignore")  # 忽略警告

TSNE_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=3000, random_state=23)


def read_pkl(path, num_list, basename):
    data = []
    for num in num_list:
        filename = num + '_' + basename
        with open((os.path.join(path, filename)), 'rb') as tf:
            _ = pickle.load(tf)
            data.append(_)
        tf.close()
    return data
    pass


def build_array(shared_lists):
    shared_array = []
    for shared_list in shared_lists:
        # for shared in shared_list:
        #     shared_array.append(shared)
        shared_array.append(shared_list)
    lists = np.array(shared_array)

    mins = lists.min(0)
    maxs = lists.max(0)
    ranges = maxs - mins
    normList = np.zeros(np.shape(lists))
    row = lists.shape[0]
    normList = lists - np.tile(mins, (row,1))
    normList = normList / np.tile(ranges, (row,1))

    return normList
    # return (lists - np.mean(lists)) / np.std(lists)


def random_index(random_range, random_num=0):
    index = [i for i in range(0, random_range)]
    random.shuffle(index)
    index = index[:random_num]
    return index


def visualization_loss(loss, loss_num, filename):
    x_axis = range(0, loss_num + 1, 8)
    plt.figure(figsize=(12, 12))
    plt.plot(loss, color='red', label=r'$\mathcal{L}_{inv}$', linewidth=5)
    plt.yticks(fontsize=45)
    plt.xticks(x_axis, fontsize=45)
    # plt.xlabel('epoch', fontsize=45)
    plt.rcParams.update({'font.size': 50})
    plt.legend()
    plt.savefig(filename)
    plt.clf()


def draw_loss():
    recorder_loss = LossRecorder((r'result_loss.tsv'), total_cv=1, total_epoch=40)
    loss_mean_list = recorder_loss.read_result_from_tsv()
    visualization_loss(loss_mean_list, 40, 'common_loss.jpg')


def visualization_consistency(key_list, data, filename):
    color = ['g', 'b', 'c', 'r', 'm', 'y']
    plt.figure(figsize=(12, 12))
    p = []

    for i, key in enumerate(key_list):
        # plt.scatter(x=data[key][:, 0], y=data[key][:, 1], c=color[i], marker='o', s=90, label=translate(key))
        p.append(plt.scatter(x=data[key][:, 0], y=data[key][:, 1], c=color[i], marker='o', s=90))

    plt.yticks(np.arange(0, 0.5 + 1, 0.5), fontsize=45)
    plt.xticks(np.arange(0, 0.5 + 1, 0.5), fontsize=45)
    plt.rcParams.update({'font.size': 25})

    # plt.legend(loc='best')
    a = plt.legend(p[:(len(p) // 2)], [translate(key) for key in key_list[:(len(p) // 2)]], loc=1)
    plt.legend(p[(len(p) // 2):], [translate(key) for key in key_list[(len(p) // 2):]], loc=2)
    plt.gca().add_artist(a)

    plt.savefig(filename)
    plt.clf()

def translate(key):
    if key == 'azz':
        return '$H\'${a}'
    elif key == 'avz':
        return '$H\'${a,v}'
    elif key == 'azl':
        return '$H\'${a,t}'
    elif key == 'zvz':
        return '$H\'${v}'
    elif key == 'zvl':
        return '$H\'${v,t}'
    elif key == 'zzl':
        return '$H\'${t}'
    if key == '0':
        return 'happy'
    if key == '1':
        return 'angry'
    if key == '2':
        return 'sad'
    if key == '3':
        return 'neutral'

def draw_consistency_feature(path, num_list):
    consistent_dict_condition = {
        "azz": [],
        "zvz": [],
        "zzl": [],
        "avz": [],
        "azl": [],
        "zvl": [],
    }
    num_len = len(num_list)
    part_names = ['azz', 'avz', 'azl', 'zvl', 'zvz', 'zzl']

    consistent_feat_list = read_pkl(path=path, num_list=num_list[:num_len], basename='consistent_feat.pkl')
    miss_type_list = read_pkl(path=path, num_list=num_list[:num_len], basename='miss_type.pkl')

    consistent_feats = []
    miss_types = []

    for i in range(0, num_len):
        for item in consistent_feat_list[i].cpu().detach().numpy():
            consistent_feats.append(item)
        for item in miss_type_list[i]:
            miss_types.append(item)

    consistent_feats = np.array(consistent_feats)
    miss_types = np.array(miss_types)

    consistent_feats = TSNE_model.fit_transform(consistent_feats)
    for part_name in part_names:
        index = np.where(miss_types == part_name)
        consistent_dict_condition[part_name] = consistent_feats[index]
    for part_name in part_names:
        # normalization
        consistent_dict_condition[part_name] = build_array(consistent_dict_condition[part_name])

        # remove points which are over limited
        if part_name in ['avz', 'zvl', 'zvz']:
            p = 0
            for i, item in enumerate(consistent_dict_condition[part_name]):
                if (part_name == 'avz' and item[1] > 0.5) or (part_name == 'zvl' and item[0] > 0.5): # or (part_name == 'zvz' and item[1] < 0.5):
                    consistent_dict_condition[part_name] = np.delete(consistent_dict_condition[part_name], i-p, 0)
                    p+=1

        # take 100 points for each condition
        length = len(consistent_dict_condition[part_name]) if len(consistent_dict_condition[part_name]) < 100 else 100
        consistent_dict_condition[part_name] = consistent_dict_condition[part_name][[i for i in range(0, length)]]
        # print(part_name, consistent_dict_condition[part_name].size // 2)

    visualization_consistency(key_list=part_names, data=consistent_dict_condition, filename='consistent_feature.jpg')


def draw_consistency_label(path, num_list):
    consistent_dict_lable = {
        "0": [],
        "1": [],
        "2": [],
        "3": []
    }
    # # num_list = ['8', '10', '18', '19', '20']
    # num_list = ['10', '18', '19', '20']
    num_len = len(num_list)
    part_names = ['0', '1', '2', '3']

    consistent_feat_list = read_pkl(path=path, num_list=num_list[:num_len], basename='consistent_feat.pkl')
    label_list = read_pkl(path=path, num_list=num_list[:num_len], basename='label.pkl')

    consistent_feats = []
    labels = []

    for i in range(0, num_len):
        for item in consistent_feat_list[i].cpu().detach().numpy():
            consistent_feats.append(item)
        for item in label_list[i].cpu().detach().numpy():
            labels.append(str(item))

    consistent_feats = np.array(consistent_feats)
    labels = np.array(labels)

    consistent_feats = TSNE_model.fit_transform(consistent_feats)
    for part_name in part_names:
        index = np.where(labels == part_name)
        consistent_dict_lable[part_name] = consistent_feats[index]
        # consistent_dict_lable[part_name] = consistent_dict_lable[part_name][[i for i in range(0, 10)]]
    for part_name in part_names:
        consistent_dict_lable[part_name] = build_array(consistent_dict_lable[part_name])
    visualization_consistency(key_list=part_names, data=consistent_dict_lable, filename='consistent_label.jpg')


if __name__ == '__main__':
    path = r'shared_image/our_IEMOCAP_block_5_run_1_V020015/3/consistent'
    num_list = ['19', '20', '21', '22', '18']
    # num_list = ['3', '20', '17', '12', '15']
    draw_loss()
    # draw_consistency_feature(path=path, num_list=num_list)
    # draw_consistency_label(path=path, num_list=num_list)
