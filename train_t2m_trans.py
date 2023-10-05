import os 
import torch
import torch.nn.functional as F
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin
from torch.distributions import Categorical
import json
import clip
from tqdm import tqdm
from einops import rearrange, repeat

import options.option_transformer as option_trans
import models.vqvae as vqvae
import utils.utils_model as utils_model
import utils.eval_trans as eval_trans
from dataset import dataset_TM_train
from dataset import dataset_TM_eval
from dataset import dataset_tokenize
import models.t2m_trans as trans
from models.t2m_trans import uniform, cosine_schedule, get_mask_subset_with_prob, prob_mask_like
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')

from utils.motion_process import recover_from_ric, recover_from_rot
import visualization.plot_3d_global as plot_3d


def tensorborad_add_video_xyz(writer, xyz, nb_iter, tag, nb_vis=4, title_batch=None, outname=None):
    # breakpoint()
    xyz = xyz[:1]
    bs, seq = xyz.shape[:2]
    xyz = xyz.reshape(bs, seq, -1, 3)
    plot_xyz = plot_3d.draw_to_batch(xyz.cpu().numpy(),title_batch, outname)
    plot_xyz =np.transpose(plot_xyz, (0, 1, 4, 2, 3)) 
    writer.add_video(tag, plot_xyz, nb_iter, fps = 20)


##### ---- Exp dirs ---- #####
args = option_trans.get_args_parser()
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
args.vq_dir= os.path.join("./dataset/KIT-ML" if args.dataname == 'kit' else "./dataset/HumanML3D", f'{args.vq_name}')
os.makedirs(args.out_dir, exist_ok = True)
os.makedirs(args.vq_dir, exist_ok = True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))


from utils.word_vectorizer import WordVectorizer
w_vectorizer = WordVectorizer('./glove', 'our_vab')
val_loader = dataset_TM_eval.DATALoader(args.dataname, False, 32, w_vectorizer, split='train_small')

dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if args.dataname == 'kit' else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'

wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)




##### ---- Network ---- #####
clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False)  # Must set jit=False for training
clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                       args.nb_code,
                       args.code_dim,
                       args.output_emb_width,
                       args.down_t,
                       args.stride_t,
                       args.width,
                       args.depth,
                       args.dilation_growth_rate)

print ('loading VQ-VAE checkpoint from {}'.format(args.resume_pth))
ckpt = torch.load(args.resume_pth, map_location='cpu')
net.load_state_dict(ckpt['net'], strict=True)
net.eval()
net.cuda()

    # ##### ---- Dataloader ---- #####
    # train_loader_token = dataset_tokenize.DATALoader(args.dataname, 1, unit_length=2**args.down_t)
    # ##### ---- get code ---- #####
    # for batch in tqdm(train_loader_token):
    #     pose, name = batch
    #     bs, seq = pose.shape[0], pose.shape[1]

    #     pose = pose.cuda().float() # bs, nb_joints, joints_dim, seq_len
    #     target = net.encode(pose)
    #     target = target.cpu().numpy()
    #     np.save(pjoin(args.vq_dir, name[0] +'.npy'), target)

trans_encoder = trans.Text2Motion_Transformer(num_vq=args.nb_code, 
                                embed_dim=args.embed_dim_gpt, 
                                clip_dim=args.clip_dim, 
                                block_size=args.block_size, 
                                num_layers=args.num_layers, 
                                n_head=args.n_head_gpt, 
                                drop_out_rate=args.drop_out_rate, 
                                fc_rate=args.ff_rate,
                                has_cross_attn= not args.no_cross_attn)

trans_encoder.train()
trans_encoder.cuda()

##### ---- Optimizer & Scheduler ---- #####
optimizer = utils_model.initial_optim(args.decay_option, args.lr, args.weight_decay, trans_encoder, args.optimizer)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

##### ---- Optimization goals ---- #####

weight_class = torch.ones((args.nb_code+1), dtype=torch.float).cuda()
weight_class[-1] = 0.1
loss_ce = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='mean', weight=weight_class)

nb_iter, avg_loss_cls, avg_acc = 0, 0., 0.
right_num = 0
nb_sample_train = 0

if args.resume_trans is not None:
    logger.info ('loading transformer checkpoint from {}'.format(args.resume_trans))
    ckpt = torch.load(args.resume_trans, map_location='cpu')
    trans_encoder.load_state_dict(ckpt['trans'], strict=True)

    if "optimizer" in ckpt.keys():
        optimizer.load_state_dict(ckpt["optimizer"])
    
    if "scheduler" in ckpt.keys():
        scheduler.load_state_dict(ckpt["scheduler"])

    if "nb_iter" in ckpt.keys():
        nb_iter = ckpt["nb_iter"]
        logger.info ('Resume training from {}'.format(nb_iter))


train_loader = dataset_TM_train.DATALoader(args.dataname, args.batch_size, args.nb_code, args.vq_name, unit_length=2**args.down_t, num_workers=8, split=args.split)
train_loader_iter = dataset_TM_train.cycle(train_loader)


# print('Start evaluation first')
##### ---- Training ---- #####
# best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = eval_trans.evaluation_transformer(
#     args.out_dir, val_loader, net, trans_encoder, logger, writer, 0, best_fid=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, clip_model=clip_model, eval_wrapper=eval_wrapper)
best_fid=1000
best_iter=0
best_div=100
best_top1=0
best_top2=0
best_top3=0
best_matching=100


# if nb_iter > 0:
#     logger.info(f'Starting training from iter {nb_iter}')

# best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = \
#         eval_trans.evaluation_transformer_debug(
#             args.out_dir, 
#             val_loader, 
#             net, 
#             trans_encoder, 
#             logger, 
#             writer, 
#             nb_iter, 
#             best_fid, 
#             best_iter, 
#             best_div, 
#             best_top1, 
#             best_top2, 
#             best_top3, 
#             best_matching, 
#             clip_model=clip_model, eval_wrapper=eval_wrapper, optimizer=optimizer, scheduler=scheduler)



print('Start training')
while nb_iter <= args.total_iter:
    # print(nb_iter)
    batch = next(train_loader_iter)
    clip_text, m_tokens, m_tokens_len, mask_token = batch
    m_tokens, m_tokens_len, mask_token = m_tokens.cuda(), m_tokens_len.cuda(), mask_token.cuda()
    
    bs, seq_len = m_tokens.shape[0], m_tokens.shape[1]
    # seq_len = seq_len -1 # FIXME -1 to remove the [END] token
    
    target = m_tokens   # (bs, 26)
    device = target.device

    
    text = clip.tokenize(clip_text, truncate=True).cuda()
    
    feat_clip_text = clip_model.encode_text(text).float()

    # breakpoint()
    input_index = target
    # mask_token = mask_token[:, :-1]


    text_mask = torch.ones_like(mask_token)
    if args.cond_drop_prob > 0:
        keep_mask = prob_mask_like((bs,), 1 - args.cond_drop_prob, device = text_mask.device)
        text_mask = rearrange(keep_mask, 'b -> b 1') & text_mask
    

    # NOTE masking
    # if args.pkeep == -1:
    #     proba = np.random.rand(1)[0]
    #     mask = torch.bernoulli(proba * torch.ones(input_index.shape,
    #                                                      device=input_index.device))
    # else:
    #     mask = torch.bernoulli(args.pkeep * torch.ones(input_index.shape,
    #                                                      device=input_index.device))
    # mask = mask.round().to(dtype=torch.int64)
    # r_indices = torch.randint_like(input_index, args.nb_code)
    # a_indices = mask*input_index+(1-mask)*r_indices
    ######################!SECTION
    
    # batch, seq_len = idxs.shape[0], idxs.shape[1]

    rand_step = torch.randint(0, 18, (bs,)).cuda() / 18.
    mask_token_prob = torch.cos(rand_step * np.pi * 0.5) # cosine schedule was best

    # FIXME scale prob to only 0.0 - > 0.25 
    # mask_token_prob = mask_token_prob * 0.5
    num_token_masked = ((m_tokens_len + 1) * mask_token_prob).round().clamp(min = 1)

    mask = get_mask_subset_with_prob(mask_token, mask_token_prob)

    a_indices = torch.where(mask, net.vqvae.num_code + 2, input_index)
    

    # breakpoint()
    # NOTE Forward model
    cls_pred = trans_encoder(a_indices, feat_clip_text, mask_token, text_mask)

    # breakpoint()
    cls_pred = cls_pred.contiguous()

    # print('debug', torch.nonzero(torch.argmax(cls_pred[-1], dim=-1)==512))
    # if nb_iter % 10 == 0:
    #     # print('debug softmax', torch.softmax(cls_pred, dim=-1)[:5,:,-1])
    #     print('debug pred token', torch.unique(torch.argmax(cls_pred, dim=-1), return_counts=True))

    loss_cls = 0.0

    cls_pred_ = cls_pred.flatten(0,1)[mask.flatten(0,1)]
    target_ = target.flatten(0,1)[mask.flatten(0,1)]

    # cls_pred_ = cls_pred.flatten(0,1)[mask_token.flatten(0,1)]
    # target_ = target[:,:-1].flatten(0,1)[mask_token.flatten(0,1)]
    loss_cls = loss_ce(cls_pred_, target_) # B, len


    probs = torch.softmax(cls_pred_, dim=-1)
    dist = Categorical(probs)
    cls_pred_index = dist.sample()

    # if nb_iter > 300:
    #     breakpoint()
    right_num += (cls_pred_index == target_).sum().item()

    ## global loss
    optimizer.zero_grad()
    loss_cls.backward()
    optimizer.step()
    scheduler.step()

    avg_loss_cls = avg_loss_cls + loss_cls.item()
    nb_sample_train = nb_sample_train + (mask).sum().item()


    # FIXME debug
                    # draw_pred = []
                    # draw_text_pred = []

                    # cls_arg = cls_pred.argmax(-1)
                    # ids = cls_arg
                    # for k in tqdm(range(4,8)):
                    #     ids_ = ids[k]
                    #     # breakpoint()
                    #     try:
                    #         first_end = torch.nonzero(ids_ == net.vqvae.num_code).view(-1)[0]
                    #     except:
                    #         first_end = -1
                    #     # print('first_end', first_end)
                    #     ids_ = ids_[:first_end]
                    #     # breakpoint()
                    #     # breakpoint()
                    #     pred_pose = net.forward_decoder(ids_[None,:])

                    #     # breakpoint()
                    #     # cur_len = pred_pose.shape[1]

                    #     # pred_len[k] = min(cur_len, 196)
                    #     # # pred_len[k] = m_length[k]
                    #     # if pred_len[k] < 4:
                    #     #     continue

                    #     # # pred_pose_eval[k:k+1, :pred_len[k]] = pose.cuda()[k:k+1, :pred_len[k]]
                    #     # pred_pose_eval[k:k+1, :pred_len[k]] = pred_pose[:, :pred_len[k]]

                    #     pred_denorm = train_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
                    #     pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().cuda(), 22)

                    #     if k < 4:
                    #         draw_pred.append(pred_xyz)
                    #         draw_text_pred.append(clip_text[k])
                    

                    #     tensorborad_add_video_xyz(writer, pred_xyz, 100000, tag='./Vis/debug_pred'+str(k), nb_vis=1, title_batch=[clip_text[k]], outname=[os.path.join('./results/maskgit_debug', 'pred'+str(k)+'.gif')])
                        
                    # for k in tqdm(range(4,8)):
                    #     input_index_ = input_index[k]
                    #     # breakpoint()
                    #     try:
                    #         first_end = torch.nonzero(input_index_ == net.vqvae.num_code).view(-1)[0]
                    #     except:
                    #         first_end = -1
                    #     # print('first_end', first_end)
                    #     input_index_ = input_index_[:first_end]
                    #     # breakpoint()

                    #     gt_pose = net.forward_decoder(input_index_[None,:])

                    #     # breakpoint()
                    #     # cur_len = gt_pose.shape[1]

                    #     # gt_len[k] = min(cur_len, 196)
                    #     # # pred_len[k] = m_length[k]
                    #     # if gt_len[k] < 4:
                    #     #     continue

                    #     # pred_pose_eval[k:k+1, :pred_len[k]] = pose.cuda()[k:k+1, :pred_len[k]]
                    #     # gt_pose_eval[k:k+1, :pred_len[k]] = gt_pose[:, :pred_len[k]

                    #     pred_denorm = train_loader.dataset.inv_transform(gt_pose.detach().cpu().numpy())
                    #     pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().cuda(), 22)

                    #     if k < 4:
                    #         draw_pred.append(pred_xyz)
                    #         draw_text_pred.append(clip_text[k])
                    #     tensorborad_add_video_xyz(writer, pred_xyz, 100000, tag='./Vis/debug_gt'+str(k), nb_vis=1, title_batch=[clip_text[k]], outname=[os.path.join('./results/maskgit_debug', 'gt'+str(k)+'.gif')])

                    # quit()


    nb_iter += 1
    if nb_iter % args.print_iter ==  0 :
        avg_loss_cls = avg_loss_cls / args.print_iter
        avg_acc = right_num * 100 / nb_sample_train
        writer.add_scalar('./Loss/train', avg_loss_cls, nb_iter)
        writer.add_scalar('./ACC/train', avg_acc, nb_iter)
        msg = f"Train. Iter {nb_iter} : Loss. {avg_loss_cls:.5f}, ACC. {avg_acc:.4f}"
        logger.info(msg)
        avg_loss_cls = 0.
        right_num = 0
        nb_sample_train = 0

    if nb_iter % args.eval_iter ==  0:
        best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = \
        eval_trans.evaluation_transformer(
            args.out_dir, 
            val_loader, 
            net, 
            trans_encoder, 
            logger, 
            writer, 
            nb_iter, 
            best_fid, 
            best_iter, 
            best_div, 
            best_top1, 
            best_top2, 
            best_top3, 
            best_matching, 
            clip_model=clip_model, eval_wrapper=eval_wrapper, optimizer=optimizer, scheduler=scheduler)


    if nb_iter % 1000 == 0:
        
        save_path = os.path.join(args.out_dir, 'net_last.pth')
        if os.path.isfile(save_path):
            os.remove(save_path)
        
        torch.save({'trans' : trans_encoder.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'nb_iter': nb_iter}, save_path)
        
        new_save_path = os.path.join(args.out_dir, f'net_{nb_iter}.pth')
        copy_cmd = f'cp "{save_path}" "{new_save_path}"'
        os.system(copy_cmd)
        # torch.save({'trans' : trans.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, os.path.join(out_dir, f'net_{nb_iter}.pth'))


    if nb_iter == args.total_iter: 
        msg_final = f"Train. Iter {best_iter} : FID. {best_fid:.5f}, Diversity. {best_div:.4f}, TOP1. {best_top1:.4f}, TOP2. {best_top2:.4f}, TOP3. {best_top3:.4f}"
        logger.info(msg_final)
        break            