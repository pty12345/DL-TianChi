if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

export CUDA_VISIBLE_DEVICES=4

python -u main.py \
    --model MLP \
    --enc_in 15 \
    --seq_len 30 \
    --pred_len 10 \
    --batch_size 32 \
    --learning_rate 0.0003 \
    --train_epochs 100 \
    --patience 5 >logs/train.log