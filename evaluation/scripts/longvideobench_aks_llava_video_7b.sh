base_score_path=./selected_frames/longvideobench/blip
score_type=selected_frames
dataset_name=longvideobench

python ./evaluation/change_score.py \
    --base_score_path $base_score_path \
    --score_type $score_type \
    --dataset_name $dataset_name 

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 --main_process_port 12345 -m lmms_eval \
    --model llava_vid \
    --model_args pretrained=./checkpoints/llava_video_7b,conv_template=chatml_direct,video_decode_backend=decord,max_frames_num=64,overwrite=False,use_topk=True \
    --tasks longvideobench_val_v \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llavavid_7b_qwen_lvb_v \
    --output_path ./results/${score_type}
