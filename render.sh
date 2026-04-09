#!/bin/bash

# IDS="75 135 155 203 209 243"
IDS="100"

# IDS="76"

#MODEL=runs/data=human-ml3d/motion_model=movit-full-timemask/optim=contrastive-fixed/text_model=clip/data_rep=cont_6d_plus_rifke/space-dim=256/run-0
#MODEL=runs/data=human-ml3d/motion_model=bidir-gru/optim=info-nce/text_model=bert-lstm/data_rep=cont_6d_plus_rifke/space-dim=256/run-0
MODEL=runs/data=human-ml3d/motion_model=upper-lower-gru/optim=info-nce/text_model=bert-lstm/data_rep=cont_6d_plus_rifke/space-dim=256/run-0
#MODEL=runs/data=human-ml3d/motion_model=dgstgcn/optim=info-nce/text_model=clip/data_rep=cont_6d_plus_rifke/space-dim=256/run-0
#MODEL=runs/data=human-ml3d/motion_model=bidir-gru/optim=info-nce/text_model=clip/data_rep=cont_6d_plus_rifke/space-dim=256/run-0
#MODEL=runs/data=human-ml3d/motion_model=movit-full-timemask/optim=info-nce/text_model=clip/data_rep=cont_6d_plus_rifke/space-dim=256/run-0
#MODEL=runs/data=human-ml3d/motion_model=upper-lower-gru/optim=info-nce/text_model=clip/data_rep=cont_6d_plus_rifke/space-dim=256/run-0

#CUDA_VISIBLE_DEVICES="" conda run --no-capture-output -n t2m python render.py MODEL --best_on_metric all --set test --query_ids_to_render $IDS --override_existing_videos
# CUDA_VISIBLE_DEVICES="" conda run --no-capture-output -n t2m python render.py --run $MODEL --best_on_metric all --set test --query_ids_to_render 75 135 155 203 209 243 --override_existing_videos
CUDA_VISIBLE_DEVICES="" conda run --no-capture-output -n t2m python render.py --run $MODEL --best_on_metric all --set test --query_ids_to_render 100 --override_existing_videos


for i in $IDS
do
    conda run --no-capture-output -n t2m python make_mosaic_video.py outputs/renders/test/$i
done
