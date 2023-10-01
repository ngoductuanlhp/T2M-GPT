CUDA_VISIBLE_DEVICES=$1 python3 VQ_eval.py \
--batch-size 256 \
--lr 2e-4 \
--total-iter 300000 \
--lr-scheduler 200000 \
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
--exp-name TEST_VQVAE \
--resume-pth output/VQVAE_loc3d_train_mini/net_75000.pth


# --width 128 \
# --output-emb-width 128 \
# --code-dim 128 \