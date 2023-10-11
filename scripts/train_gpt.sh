CUDA_VISIBLE_DEVICES=$1 python3 train_t2m_trans.py  \
--exp-name MaskGIT_t5_embedding2_separatepredlength \
--batch-size 32 \
--num-layers 15 \
--clip-dim 768 \
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
--total-iter 800000 \
--lr-scheduler 500000 \
--lr 0.00005 \
--dataname t2m \
--split train \
--down-t 2 \
--depth 3 \
--quantizer ema_reset \
--eval-iter 40000 \
--pkeep 0.5 \
--dilation-growth-rate 3 \
--vq-act relu 
# \
# --resume-trans output/GPT_MaskGit_classifierfree_predlength_15layers_t5_embedding/net_1000.pth