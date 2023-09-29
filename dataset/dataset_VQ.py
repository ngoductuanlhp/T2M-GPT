import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
from utils.motion_process import recover_from_rot, recover_root_rot_pos, quaternion_to_cont6d
import os

class VQMotionDataset(data.Dataset):
    def __init__(self, dataset_name, window_size = 64, unit_length = 4):
        self.window_size = window_size
        self.unit_length = unit_length
        self.dataset_name = dataset_name

        if dataset_name == 't2m':
            self.data_root = './dataset/HumanML3D'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            self.max_motion_length = 196
            self.meta_dir = 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'

        elif dataset_name == 'kit':
            self.data_root = './dataset/KIT-ML'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 21

            self.max_motion_length = 196
            self.meta_dir = 'checkpoints/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
        
        joints_num = self.joints_num

        # if not os.path.exists(pjoin(self.motion_dir, 'mean_rot6d.npy')):
        #     self.mean_variance(self.motion_dir, self.motion_dir, self.joints_num)

        mean = np.load(pjoin(self.data_root, 'Mean.npy'))[..., :(self.joints_num - 1) * 3 + 4]
        std = np.load(pjoin(self.data_root, 'Std.npy'))[..., :(self.joints_num - 1) * 3 + 4]

        split_file = pjoin(self.data_root, 'train.txt')

        self.data = []
        self.lengths = []
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        for name in tqdm(id_list):
            # try:
            motion = np.load(pjoin(self.motion_dir, name + '.npy'))
            # motion = self.preprocess(motion)
            # np.save(pjoin(self.data_root, 'new_joint_vecs', name + '.npy'), motion)

            if motion.shape[0] < self.window_size:
                continue

            self.lengths.append(motion.shape[0] - self.window_size)
            self.data.append(motion)

                # breakpoint()
            # except:
            #     # Some motion may not exist in KIT dataset
            #     pass

        self.mean = mean
        self.std = std
        print("Total number of motions {}".format(len(self.data)))

    def inv_transform(self, data):
        return data * self.std + self.mean
    
    def compute_sampling_prob(self) : 
        
        prob = np.array(self.lengths, dtype=np.float32)
        prob /= np.sum(prob)
        return prob
    
    def __len__(self):
        return len(self.data)
    

    def mean_variance(self, data_dir, save_dir, joints_num):
        file_list = os.listdir(data_dir)
        data_list = []

        for file in tqdm(file_list):
            data = np.load(pjoin(data_dir, file))
            
            if np.isnan(data).any():
                print(file)
                continue

            data = self.preprocess(data)
            data_list.append(data)

        data = np.concatenate(data_list, axis=0)
        print(data.shape)
        Mean = data.mean(axis=0)
        Std = data.std(axis=0)
        Std[0:3] = Std[0:3].mean() / 1.0
        Std[3:] = Std[3:].mean() / 1.0
        # Std[3:4] = Std[3:4].mean() / 1.0
        # Std[4: 4+(joints_num - 1) * 3] = Std[4: 4+(joints_num - 1) * 3].mean() / 1.0
        # Std[4+(joints_num - 1) * 3: 4+(joints_num - 1) * 9] = Std[4+(joints_num - 1) * 3: 4+(joints_num - 1) * 9].mean() / 1.0
        # Std[4+(joints_num - 1) * 9: 4+(joints_num - 1) * 9 + joints_num*3] = Std[4+(joints_num - 1) * 9: 4+(joints_num - 1) * 9 + joints_num*3].mean() / 1.0
        # Std[4 + (joints_num - 1) * 9 + joints_num * 3: ] = Std[4 + (joints_num - 1) * 9 + joints_num * 3: ].mean() / 1.0

        # assert 8 + (joints_num - 1) * 9 + joints_num * 3 == Std.shape[-1]

        np.save(pjoin(save_dir, 'Mean_rot6d.npy'), Mean)
        np.save(pjoin(save_dir, 'Std_rot6d.npy'), Std)

        return Mean, Std
    
    # def preprocess(self, data):
    #     data = torch.from_numpy(data)
    #     r_rot_quat, r_pos = recover_root_rot_pos(data)
    #     positions = data[..., :(self.joints_num - 1) * 3 + 4]

    #     # r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

    #     # start_indx = 1 + 2 + 1 + (self.joints_num - 1) * 3
    #     # end_indx = start_indx + (self.joints_num - 1) * 6
    #     # cont6d_params = data[:, start_indx:end_indx]
    #     # #     print(r_rot_cont6d.shape, cont6d_params.shape, r_pos.shape)
    #     # cont6d_params = torch.cat([r_pos, r_rot_cont6d, cont6d_params], dim=-1)
    #     # cont6d_params = cont6d_params.reshape(-1, self.joints_num, 6)
    #     # print('after', cont6d_params.shape)
    #     return cont6d_params.numpy()


    def __getitem__(self, item):
        motion = self.data[item]
        motion = motion[..., :(self.joints_num - 1) * 3 + 4]

        # motion = motion[:, ]
        # print('before', motion.shape)

        # motion = self.preprocess(motion)
        
        idx = random.randint(0, len(motion) - self.window_size)

        motion = motion[idx:idx+self.window_size]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion

def DATALoader(dataset_name,
               batch_size,
               num_workers = 8,
               window_size = 64,
               unit_length = 4):
    
    trainSet = VQMotionDataset(dataset_name, window_size=window_size, unit_length=unit_length)
    prob = trainSet.compute_sampling_prob()
    sampler = torch.utils.data.WeightedRandomSampler(prob, num_samples = len(trainSet) * 1000, replacement=True)
    train_loader = torch.utils.data.DataLoader(trainSet,
                                              batch_size,
                                              shuffle=True,
                                              #sampler=sampler,
                                              num_workers=num_workers,
                                              #collate_fn=collate_fn,
                                              drop_last = True)
    
    return train_loader

def cycle(iterable):
    while True:
        for x in iterable:
            yield x
