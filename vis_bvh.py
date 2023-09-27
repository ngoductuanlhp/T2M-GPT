
import os
import json
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import models.vqvae as vqvae
import options.option_vq as option_vq
import utils.utils_model as utils_model
from dataset import dataset_TM_eval
import utils.eval_trans as eval_trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from utils.motion_process import recover_from_rot, recover_root_rot_pos, quaternion_to_cont6d

from utils.skeleton import Skeleton
from utils.paramUtil import *
import visualization.plot_3d_global as plot_3d

from tqdm import tqdm

vis_dir = '/home/tuan/tdngo/motion_ws/T2M-GPT/results/mixamo_rot6d/Mousey_m'
motion_dir = '/home/tuan/tdngo/motion_ws/deep-motion-editing/datasets/Mixamo_rot6d/Mousey_m'

motion_files = sorted(os.listdir(motion_dir))

for f in tqdm(motion_files):
    file_dict = np.load(os.path.join(motion_dir, f), allow_pickle=True)

    # breakpoint()
    offsets = file_dict['offsets']
    kinematic_chains = file_dict['kinematic_chains']
    positions = torch.from_numpy(file_dict['positions'])
    rotations_6d = torch.from_numpy(file_dict['rotations_6d'])

    new_offsets = [[0,0,0]]
    for o in offsets:
        new_offsets.append(o)
    offsets = new_offsets
    offsets = np.array(offsets)
    skeleton = Skeleton(offset=torch.from_numpy(offsets), 
                        kinematic_tree=kinematic_chains,
                        device='cpu')
    skeleton.set_offset(torch.from_numpy(offsets))

    positions = skeleton.forward_kinematics_cont6d(rotations_6d, positions)
    positions = positions.reshape(1, -1, 28, 3)
    positions = positions / 60.0

    pose_vis = plot_3d.draw_to_batch(positions.cpu().numpy(), 'debug', [os.path.join(vis_dir, f.replace('npz', 'gif'))], chain=kinematic_chains)