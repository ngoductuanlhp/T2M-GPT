import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import utils.paramUtil as paramUtil
from torch.utils.data._utils.collate import default_collate


def collate_fn(batch):
    # batch.sort(key=lambda x: x[3], reverse=True)

    t5_embedding_list = []
    m_tokens_list = []
    m_tokens_len_list = []
    mask_token_list = []

    max_len_embed = -1
    for b in batch:
        t5_embedding, m_tokens, m_tokens_len, mask_token = b
        t5_embedding_list.append(t5_embedding)
        m_tokens_list.append(torch.from_numpy(m_tokens))
        m_tokens_len_list.append(m_tokens_len)
        mask_token_list.append(torch.from_numpy(mask_token))

        max_len_embed = max(max_len_embed, t5_embedding.shape[1])

    t5_embedding_tensor = torch.zeros((len(batch), max_len_embed, 768), dtype=torch.float)
    t5_embedding_mask = torch.zeros((len(batch), max_len_embed), dtype=torch.bool)

    for b, embed in enumerate(t5_embedding_list):
        t5_embedding_tensor[b, :embed.shape[1]] = embed
        t5_embedding_mask[b, :embed.shape[1]] = 1

    m_tokens_list = torch.stack(m_tokens_list, dim=0)
    mask_token_list = torch.stack(mask_token_list, dim=0)
    m_tokens_len_list = torch.tensor(m_tokens_len_list)
    return t5_embedding_tensor, t5_embedding_mask, m_tokens_list, m_tokens_len_list, mask_token_list


'''For use of training text-2-motion generative model'''
class Text2MotionDataset(data.Dataset):
    def __init__(self, dataset_name, feat_bias = 5, unit_length = 4, codebook_size = 1024, tokenizer_name=None, split='train'):
        
        self.max_length = 64
        self.pointer = 0
        self.dataset_name = dataset_name

        self.unit_length = unit_length
        # self.mot_start_idx = codebook_size
        # self.mot_end_idx = codebook_size
        self.mot_pad_idx = codebook_size + 1
        if dataset_name == 't2m':
            self.data_root = './dataset/HumanML3D'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            radius = 4
            fps = 20
            self.max_motion_length = 26 if unit_length == 8 else 51
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
            self.max_motion_length = 26 if unit_length == 8 else 51
            kinematic_chain = paramUtil.kit_kinematic_chain
            self.meta_dir = 'checkpoints/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'

        split_file = pjoin(self.data_root, f'{split}.txt')

        self.mean = np.load(pjoin(self.meta_dir, 'mean.npy'))
        self.std = np.load(pjoin(self.meta_dir, 'std.npy'))


        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        data_dict = {}
        for name in tqdm(id_list):
            try:
                m_token_list = np.load(pjoin(self.data_root, tokenizer_name, '%s.npy'%name))
                flan_t5_embedding_list = torch.load(pjoin(self.data_root, 'flan-t5-base_text_embeddings', '%s.pth'%name), map_location='cpu')
                flan_t5_embedding_list = [f.detach() for f in flan_t5_embedding_list]
                # breakpoint()
                # Read text
                with cs.open(pjoin(self.text_dir, name + '.txt')) as f:
                    text_data = []
                    flag = False
                    lines = f.readlines()

                    for line in lines:
                        try:
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            t_tokens = line_split[1].split(' ')
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict['caption'] = caption
                            text_dict['tokens'] = t_tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                m_token_list_new = [tokens[int(f_tag*fps/unit_length) : int(to_tag*fps/unit_length)] for tokens in m_token_list if int(f_tag*fps/unit_length) < int(to_tag*fps/unit_length)]

                                if len(m_token_list_new) == 0:
                                    continue
                                new_name = '%s_%f_%f'%(name, f_tag, to_tag)

                                data_dict[new_name] = {'m_token_list': m_token_list_new,
                                                       'text':[text_dict],
                                                       'flan_t5_embedding': flan_t5_embedding_list}
                                new_name_list.append(new_name)
                        except:
                            pass

                if flag:
                    data_dict[name] = {'m_token_list': m_token_list,
                                       'text':text_data,
                                       'flan_t5_embedding': flan_t5_embedding_list}
                    new_name_list.append(name)
            except:
                pass
        self.data_dict = data_dict
        self.name_list = new_name_list

    def __len__(self):
        return len(self.data_dict)
    

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __getitem__(self, item):

        # print('0')
        data = self.data_dict[self.name_list[item]]

        # print(data.keys())
        m_token_list, t5_embedding_list = data['m_token_list'], data['flan_t5_embedding'] # data['text']
        m_tokens = random.choice(m_token_list)

        # print(len(m_token_list), m_token_list)

        t5_embedding = random.choice(t5_embedding_list)
        # t5_embedding = t5_embedding['caption']
        # t5_embedding = text_data['flan_t5_embedding']

        # print('1')
        
        coin = np.random.choice([False, False, True])
        # print(len(m_tokens))
        if coin:
            # drop one token at the head or tail
            coin2 = np.random.choice([True, False])
            if coin2:
                m_tokens = m_tokens[:-1]
            else:
                m_tokens = m_tokens[1:]
        m_tokens_len = m_tokens.shape[0]


        # print('2')

        # if m_tokens_len+1 < self.max_motion_length:
        #     m_tokens = np.concatenate([m_tokens, np.ones((1), dtype=int) * self.mot_end_idx, np.ones((self.max_motion_length-1-m_tokens_len), dtype=int) * self.mot_pad_idx], axis=0)
        # else:
        #     # quit()
        #     m_tokens = np.concatenate([m_tokens, np.ones((1), dtype=int) * self.mot_end_idx], axis=0)

        m_tokens = np.concatenate([m_tokens, np.ones((self.max_motion_length-m_tokens_len), dtype=int) * self.mot_pad_idx], axis=0)


        mask_token = np.ones((m_tokens.shape[0]), dtype=bool)
        mask_token[m_tokens_len:] = 0

        m_tokens = m_tokens.reshape(-1)

        # print('dataloader', t5_embedding.shape)
        return t5_embedding, m_tokens, m_tokens_len, mask_token




def DATALoader(dataset_name,
                batch_size, codebook_size, tokenizer_name, unit_length=4,
                num_workers = 8, split='train') : 

    train_loader = torch.utils.data.DataLoader(Text2MotionDataset(dataset_name, codebook_size = codebook_size, tokenizer_name = tokenizer_name, unit_length=unit_length, split=split),
                                              batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn,
                                              drop_last = True)
    

    return train_loader


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


