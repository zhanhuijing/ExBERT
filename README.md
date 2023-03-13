Download dataset from google drive:
https://drive.google.com/file/d/1Cgn-42MJD-p6JNiJ0HHpuL_nyf7MEjrH/view?usp=sharing

Then you can run the cmd: 
CUDA_VISIBLE_DEVICES=0 python -u main.py \
--data_path ./TripAdvisor/reviews_profile_index.pickle \
--index_dir ./TripAdvisor/1/ \
--cuda \
--checkpoint ./TA_output/TA_1_nsp_clsf_10_reg_0.8_gamma_0.8_nomasked/ \
--image_fea_path None \
--key_len 4 \
--epochs_teacher 200 \
--epochs_student 200 \
--peter_mask \
--profile_len 5 \
--feature_extract_path None \
--user_profile_path ./TripAdvisor/user_profile.json \
--item_profile_path ./TripAdvisor/item_profile.json \
--batch_size 128 \
--item_keyword_path None \
--endure_times 10 \
--teacher_model_path None \
--words 15 \
--clsf_reg 10 \
--neg_rate 0.8 \
--gamma 0.8 \
--test_step 5 \
--log_interval 500  >> ./TA_output/TA_1_nsp_clsf_10_reg_0.8_gamma_0.8_nomasked.log



