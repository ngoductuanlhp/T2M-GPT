import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm

import utils.paramUtil as paramUtil
from torch.utils.data._utils.collate import default_collate

from utils.motion_process import recover_from_rot, recover_root_rot_pos, quaternion_to_cont6d
import os

def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


'''For use of training text-2-motion generative model'''
class Text2MotionDataset(data.Dataset):
    def __init__(self, dataset_name, is_test, w_vectorizer, feat_bias = 5, max_text_len = 20, unit_length = 4):
        
        self.max_length = 20
        self.pointer = 0
        self.dataset_name = dataset_name
        self.is_test = is_test
        self.max_text_len = max_text_len
        self.unit_length = unit_length
        self.w_vectorizer = w_vectorizer
        if dataset_name == 't2m':
            self.data_root = './dataset/HumanML3D'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            radius = 4
            fps = 20
            self.max_motion_length = 196
            dim_pose = 263
            kinematic_chain = paramUtil.t2m_kinematic_chain
            self.meta_dir = 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
        elif dataset_name == 'kit':
            self.data_root = './dataset/KIT-ML'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 21
            radius = 240 * 8
            fps = 12.5
            dim_pose = 251
            self.max_motion_length = 196
            kinematic_chain = paramUtil.kit_kinematic_chain
            self.meta_dir = 'checkpoints/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'


        joints_num = self.joints_num
        self.start_ind = 1 + 2 + 1 + (joints_num - 1) * 3
        self.end_ind = self.start_ind + (joints_num - 1) * 6
        # mean = np.load(pjoin(self.meta_dir, 'mean.npy'))
        # std = np.load(pjoin(self.meta_dir, 'std.npy'))
        # mean = np.load(pjoin(self.data_root, 'Mean_rot6d.npy'))
        # std = np.load(pjoin(self.data_root, 'Std_rot6d.npy'))
        mean = np.load(pjoin(self.data_root, 'Mean.npy'))
        mean = np.concatenate([mean[:4], mean[self.start_ind:self.end_ind]], axis=-1)

        std = np.load(pjoin(self.data_root, 'Std.npy'))
        std = np.concatenate([std[:4], std[self.start_ind:self.end_ind]], axis=-1)

        
        if is_test:
            split_file = pjoin(self.data_root, 'test.txt')
        else:
            split_file = pjoin(self.data_root, 'val.txt')

        split_file = pjoin(self.data_root, 'train_small.txt')

        min_motion_len = 40 if self.dataset_name =='t2m' else 24
        # min_motion_len = 64


        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(self.motion_dir, name + '.npy'))
                motion = np.concatenate([motion[:, :4], motion[:, self.start_ind:self.end_ind]], axis=-1)

                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue

                # motion = self.preprocess(motion)

                text_data = []
                flag = False
                with cs.open(pjoin(self.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*fps) : int(to_tag*fps)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except Exception as e:
                # print(e)
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean
    

    def inv_transform_torch(self, data):
        return data * torch.from_numpy(self.std).to(data.device) + torch.from_numpy(self.mean).to(data.device)

    def forward_transform(self, data):
        return (data - self.mean) / self.std
    
    def preprocess(self, data):
        data = torch.from_numpy(data)
        r_rot_quat, r_pos = recover_root_rot_pos(data)

        r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

        start_indx = 1 + 2 + 1 + (self.joints_num - 1) * 3
        end_indx = start_indx + (self.joints_num - 1) * 6
        cont6d_params = data[:, start_indx:end_indx]
        #     print(r_rot_cont6d.shape, cont6d_params.shape, r_pos.shape)
        cont6d_params = torch.cat([r_pos, r_rot_cont6d, cont6d_params], dim=-1)
        # cont6d_params = cont6d_params.reshape(-1, self.joints_num, 6)
        # print('after', cont6d_params.shape)
        return cont6d_params.numpy()

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        name = self.name_list[idx]
        data = self.data_dict[name]
        # data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if self.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), name




def DATALoader(dataset_name, is_test,
                batch_size, w_vectorizer,
                num_workers = 8, unit_length = 4) : 
    
    val_loader = torch.utils.data.DataLoader(Text2MotionDataset(dataset_name, is_test, w_vectorizer, unit_length=unit_length),
                                              batch_size,
                                              shuffle = True,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn,
                                              drop_last = True)
    return val_loader


def cycle(iterable):
    while True:
        for x in iterable:
            yield x
