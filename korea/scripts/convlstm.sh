export CUDA_VISIBLE_DEVICES=1
nohup python train_Korea.py --model="convlstm" --device=0 --seed=1 --input_data="gdaps_kim" \
                --num_epochs=50  --normalization \
                --rain_thresholds 0.1 10.0 \
                --start_lead_time 6 --end_lead_time 88 \
                --interpolate_aws \
                --intermediate_test \
                --log_dir logs/logs_1019_Korea \
                --batch_size 1 \
                --window_size 6 \
                --dataset_dir nims \
                --loss ce+mse \
                --custom_name="Korea_convlstm_smile_20ep_seed_0_SMILE" &