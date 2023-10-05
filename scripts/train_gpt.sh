CUDA_VISIBLE_DEVICES=$1 python3 train_t2m_trans.py  \
--exp-name GPT_MaskGit_classifierfree_new_vae_token \
--batch-size 128 \
--num-layers 18 \
--embed-dim-gpt 1024 \
--nb-code 512 \
--n-head-gpt 16 \
--block-size 51 \
--ff-rate 4 \
--drop-out-rate 0.1 \
--cond-drop-prob 0.25 \
--resume-pth pretrained/VQVAE/net_last.pth \
--vq-name VQVAE \
--out-dir output \
--total-iter 300000 \
--lr-scheduler 150000 \
--lr 0.0001 \
--dataname t2m \
--split train \
--down-t 2 \
--depth 3 \
--quantizer ema_reset \
--eval-iter 5000 \
--pkeep 0.5 \
--dilation-growth-rate 3 \
--vq-act relu 
# \
# --resume-trans output/GPT_MaskGit_classifierfree_debug/net_5000.pth