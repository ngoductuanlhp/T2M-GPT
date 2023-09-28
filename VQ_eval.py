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
from utils.metrics import MRMetrics


##### ---- Exp dirs ---- #####
args = option_vq.get_args_parser()
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok = True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))


from utils.word_vectorizer import WordVectorizer
w_vectorizer = WordVectorizer('./glove', 'our_vab')


dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if args.dataname == 'kit' else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'

# wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
# eval_wrapper = EvaluatorModelWrapper(wrapper_opt)


##### ---- Dataloader ---- #####
args.nb_joints = 21 if args.dataname == 'kit' else 22

val_loader = dataset_TM_eval.DATALoader(args.dataname, True, 1, w_vectorizer, unit_length=2**args.down_t)

##### ---- Network ---- #####
net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                       args.nb_code,
                       args.code_dim,
                       args.output_emb_width,
                       args.down_t,
                       args.stride_t,
                       args.width,
                       args.depth,
                       args.dilation_growth_rate,
                       args.vq_act,
                       args.vq_norm)

if args.resume_pth : 
    logger.info('loading checkpoint from {}'.format(args.resume_pth))
    ckpt = torch.load(args.resume_pth, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=True)
net.train()
net.cuda()

# skeleton = Skeleton(offset=torch.from_numpy(t2m_raw_offsets), 
#                     kinematic_tree=t2m_kinematic_chain,
#                     device='cuda')
# skeleton.set_offset(torch.from_numpy(t2m_raw_offsets))
file_dict = np.load('dataset/Mixamo_rot6d/Mousey_m/Box Turn.npz', allow_pickle=True)
offsets = file_dict['offsets']
kinematic_chains = file_dict['kinematic_chains']
new_offsets = [[0,0,0]]
for o in offsets:
    new_offsets.append(o)
offsets = new_offsets
offsets = np.array(offsets)

skeleton = Skeleton(offset=torch.from_numpy(offsets), 
                        kinematic_tree=kinematic_chains,
                        device='cuda')
skeleton.set_offset(torch.from_numpy(offsets))

# fid = []
# div = []
# top1 = []
# top2 = []
# top3 = []
# matching = []
# repeat_time = 20

mr_metric = MRMetrics(22, force_in_meter=False)
for b,batch in tqdm(enumerate(val_loader)):
    # word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token, name = batch
    motion, m_length, name = batch

    motion = motion.cuda()
    # et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
    bs, seq = motion.shape[0], motion.shape[1]

    num_joints = 28
    
    pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).cuda()

    # print(b)
    for i in range(bs):
        # pose = val_loader.dataset.inv_transform(motion[i:i+1, :m_length[i], :].detach().cpu().numpy())
        pose = motion[i:i+1, :m_length[i], :].detach()

        # breakpoint()
        r_pos_pose = pose[..., :3]
        cont6d_params_pose = pose[..., 3:].view(-1, num_joints, 6)
        
        positions = skeleton.forward_kinematics_cont6d(cont6d_params_pose, r_pos_pose)
        # positions = positions.reshape(1, -1, 22, 3)
        positions = positions.reshape(1, -1, 28, 3)
        positions = positions / 60.0
        # positions = positions / 6.0
        
        # pose_xyz = recover_from_ric(torch.from_numpy(pose).float().cuda(), num_joints)

        pred_pose, loss_commit, perplexity = net(motion[i:i+1, :m_length[i]])
        pred_pose = pred_pose.detach()

        # breakpoint()
        r_pos_pred_pose = pred_pose[..., :3]
        cont6d_params_pred_pose = pred_pose[..., 3:].view(-1, num_joints, 6)
        # pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
        pred_positions = skeleton.forward_kinematics_cont6d(cont6d_params_pred_pose, r_pos_pred_pose)
        # pred_positions = pred_positions.reshape(1, -1, 22, 3)
        pred_positions = pred_positions.reshape(1, -1, 28, 3)
        pred_positions = pred_positions / 60.0

        mr_metric.update(pred_positions, positions, [m_length[i]])
        # self.MPJPE += torch.sum(
        #     calc_mpjpe(rst[i], ref[i], align_inds=align_inds))
        # self.PAMPJPE += torch.sum(calc_p

        pose_vis = plot_3d.draw_to_batch(pred_positions.cpu().numpy(), name, [f'./results/vq_vae_mousey_m_freeze_train/{b}_pred.gif'], chain=kinematic_chains)
        pose_vis = plot_3d.draw_to_batch(positions.cpu().numpy(), name, [f'./results/vq_vae_mousey_m_freeze_train/{b}_gt.gif'], chain=kinematic_chains)



final_metric = mr_metric.compute(None)
for k, v in final_metric.items():
    print(f'Metric {k}: {v}')
        # pred_positions = pred_positions /6.0

        # print('debug')
        
#     best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = eval_trans.evaluation_vqvae(args.out_dir, val_loader, net, logger, writer, 0, best_fid=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, eval_wrapper=eval_wrapper, draw=False, save=False, savenpy=(i==0))
#     fid.append(best_fid)
#     div.append(best_div)
#     top1.append(best_top1)
#     top2.append(best_top2)
#     top3.append(best_top3)
#     matching.append(best_matching)
# print('final result:')
# print('fid: ', sum(fid)/repeat_time)
# print('div: ', sum(div)/repeat_time)
# print('top1: ', sum(top1)/repeat_time)
# print('top2: ', sum(top2)/repeat_time)
# print('top3: ', sum(top3)/repeat_time)
# print('matching: ', sum(matching)/repeat_time)

# fid = np.array(fid)
# div = np.array(div)
# top1 = np.array(top1)
# top2 = np.array(top2)
# top3 = np.array(top3)
# matching = np.array(matching)
# msg_final = f"FID. {np.mean(fid):.3f}, conf. {np.std(fid)*1.96/np.sqrt(repeat_time):.3f}, Diversity. {np.mean(div):.3f}, conf. {np.std(div)*1.96/np.sqrt(repeat_time):.3f}, TOP1. {np.mean(top1):.3f}, conf. {np.std(top1)*1.96/np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2)*1.96/np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3)*1.96/np.sqrt(repeat_time):.3f}, Matching. {np.mean(matching):.3f}, conf. {np.std(matching)*1.96/np.sqrt(repeat_time):.3f}"
# logger.info(msg_final)