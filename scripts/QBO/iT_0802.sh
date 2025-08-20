# echo "====== 本次运行参数 ======"
# echo "jac_init: $1"

# jac_init=$1
  
# jac_mean_interval=10
# task_name="test_jacmean_${1}"
model_name=iTransformer
lens=40

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/QBO/ \
  --data_path daily_1979_2023_16436_37.csv \
  --model_id test_ep30 \
  --model $model_name \
  --data QBO \
  --data_partition 0.79 0.1 0.11 \
  --fold_loc 'vali_first' \
  --target stage \
  --seq_len $lens \
  --pred_len $lens \
  --downsample 1 \
  --e_layers 4 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 1024 \
  --batch_size 16 \
  --prints 100 \
  --learning_rate 0.001 \
  --es_delta -100 \
  --itr 1 \
  --train_epochs 30 \
  --lradj type0 \
  # --jacobian \
  # --cov_mean \
  # --jac_init  $jac_init \
  # --jac_end 15127 \
  # --jac_interval $lens \

  # --jac_init 14403 12800 \
  # --jac_end  14400\
  # --jac_interval 40 \
  # 15127 13605

