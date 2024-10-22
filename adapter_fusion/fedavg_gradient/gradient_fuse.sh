config_file="lora_animal" 

CUDA_VISIBLE_DEVICES="0" python gradient_fusion.py \
    --concept_cfg="datasets/data_cfgs/MixofShow/multi-concept/object/${config_file}.json" \
    --save_path="experiments/gradient_fusion/chilloutmix/${config_file}" \
    --pretrained_models="experiments/pretrained_models/chilloutmix" \
    --optimize_textenc_iters=500 \
    --optimize_unet_iters=50
