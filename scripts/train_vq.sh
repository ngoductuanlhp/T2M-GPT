CUDA_VISIBLE_DEVICES=$1 python3 train_vq.py \
--batch-size 1024 \
--lr 6e-4 \
--total-iter 75000 \
--lr-scheduler 50000 \
--nb-code 512 \
--down-t 2 \
--depth 3 \
--dilation-growth-rate 3 \
--out-dir output \
--dataname t2m \
--vq-act relu \
--quantizer ema_reset \
--loss-vel 0.5 \
--recons-loss l1_smooth \
--exp-name VQVAE_rot6d_Znorm
# --resume-pth output/VQVAE/net_best_fid.pth