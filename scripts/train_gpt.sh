CUDA_VISIBLE_DEVICES=$1 python3 train_t2m_trans.py  \
--exp-name GPT_debug \
--batch-size 512 \
--num-layers 9 \
--embed-dim-gpt 1024 \
--nb-code 512 \
--n-head-gpt 16 \
--block-size 51 \
--ff-rate 4 \
--drop-out-rate 0.1 \
--resume-pth output/VQVAE/net_best_fid.pth \
--vq-name VQVAE \
--out-dir output \
--total-iter 50000 \
--lr-scheduler 25000 \
--lr 0.0004 \
--dataname t2m \
--down-t 2 \
--depth 3 \
--quantizer ema_reset \
--eval-iter 5000 \
--pkeep 0.5 \
--dilation-growth-rate 3 \
--vq-act relu