
export TRAINING=/data/chercheurs/agarwals/coco/ssd_inception_v2_coco_original

CUDA_VISIBLE_DEVICES=$1 python3 train.py --logtostderr --train_dir=$TRAINING --pipeline_config_path=$TRAINING/configuration.config --job_name='worker' --task_index=$1
