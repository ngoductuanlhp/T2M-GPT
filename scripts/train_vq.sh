CUDA_VISIBLE_DEVICES=$1 python3 train_vq.py \
--batch-size 64 \
--lr 6e-5 \
--total-iter 10000 \
--lr-scheduler 4000 \
--warm-up-iter 200 \
--print-iter 100 \
--window-size 32 \
--nb-code 512 \
--down-t 2 \
--depth 3 \
--dilation-growth-rate 3 \
--out-dir output \
--dataname mousey_m \
--vq-act relu \
--quantizer ema_reset \
--loss-vel 0.5 \
--recons-loss l1_smooth \
--exp-name VQVAE_mousey_m_freeze_middle \
--pretrain-pth output/VQVAE_rot6d/net_75000.pth