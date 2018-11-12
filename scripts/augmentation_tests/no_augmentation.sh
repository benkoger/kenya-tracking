trap "exit" INT TERM
trap "kill 0" EXIT

{ CUDA_VISIBLE_DEVICES=0 python /home/golden/tf-models/research/object_detection/train.py \
    --logtostderr \
    --pipeline_config_path='/home/golden/Projects_desktop/kenya-tracking/models/augmentation_tests/no_augmentation/model/faster_rcnn_resnet101_kenya_tracking.config' \
    --train_dir='/home/golden/Projects_desktop/kenya-tracking/models/augmentation_tests/no_augmentation/train' ; } &

TRAIN_PID=$!

sleep 1m

{ CUDA_VISIBLE_DEVICES=1 python /home/golden/tf-models/research/object_detection/eval.py \
 --logtostderr \
 --pipeline_config_path='/home/golden/Projects_desktop/kenya-tracking/models/augmentation_tests/no_augmentation/model/faster_rcnn_resnet101_kenya_tracking.config' \
 --checkpoint_dir='/home/golden/Projects_desktop/kenya-tracking/models/augmentation_tests/no_augmentation/train'\
 --eval_dir='/home/golden/Projects_desktop/kenya-tracking/models/augmentation_tests/no_augmentation/eval' ; } &

EVAL_PID=$!

wait -n



