#export CUDA_VISIBLE_DEVICES=0
  
model_name=NN

python -u run.py \
  --task_name coarse_graining \
  --is_training 1 \
  --root_path ./dataset/Couzin/ \
  --model_id check_bug \
  --model $model_name \
  --data Couzin \
  --c_in 18 \
  --c_out 9 \
  --des 'Exp' \
  --d_model 256 \
  --batch_size 8 \
  --prints 5 \
  --learning_rate 0.001 \
  --patience 7 \
  --itr 1 \
  --train_epochs 15 \
  --lradj type0 \
