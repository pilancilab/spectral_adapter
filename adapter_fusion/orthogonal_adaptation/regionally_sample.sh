sample_dog=1

if [ ${sample_dog} -eq 1 ]
then
  fused_model="experiments/composed_edlora/chilloutmix/animal/combined_model_base"

  keypose_condition=''
  keypose_adaptor_weight=1.0

  sketch_condition='datasets/validation_spatial_condition/multi-objects/dogA_catA_dogB.jpg'
  sketch_adaptor_weight=1.0

  context_prompt='<dogB1> <dogB2> and <catA1> <catA2> and <dogA1> <dogA2> on a playground, in school'
  context_neg_prompt='dark, low quality, low resolution'

  region1_prompt='a <dogB1> <dogB2> on a playground, in school'
  region1_neg_prompt="[${context_neg_prompt}]"
  region1='[160, 76, 505, 350]'

  region2_prompt='a <catA1> <catA2> on a playground, in school'
  region2_neg_prompt="[${context_neg_prompt}]"
  region2='[162, 370, 500, 685]'

  region3_prompt='a <dogA1> <dogA2> on a playground, in school'
  region3_neg_prompt="[${context_neg_prompt}]"
  region3='[134, 666, 512, 1005]'

  prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}|${region3_prompt}-*-${region3_neg_prompt}-*-${region3}"

  CUDA_VISIBLE_DEVICES="0" python regionally_controlable_sampling.py \
    --pretrained_model=${fused_model} \
    --sketch_adaptor_weight=${sketch_adaptor_weight}\
    --sketch_condition=${sketch_condition} \
    --keypose_adaptor_weight=${keypose_adaptor_weight}\
    --keypose_condition=${keypose_condition} \
    --save_dir="results/multi-concept/orthogonal_animal_playground" \
    --prompt="${context_prompt}" \
    --negative_prompt="${context_neg_prompt}" \
    --prompt_rewrite="${prompt_rewrite}" \
    --suffix="baseline" \
    --seed=7
  
fi

if [ ${sample_dog} -eq 1 ]
then
  fused_model="experiments/composed_edlora/chilloutmix/animal/combined_model_base"

  keypose_condition=''
  keypose_adaptor_weight=1.0

  sketch_condition='datasets/validation_spatial_condition/multi-objects/dogA_catA_dogB.jpg'
  sketch_adaptor_weight=1.0

  context_prompt='<dogB1> <dogB2> and <catA1> <catA2> and <dogA1> <dogA2> in galaxy, starwar background'
  context_neg_prompt='dark, low quality, low resolution'

  region1_prompt='a <dogB1> <dogB2> in galaxy, starwar background'
  region1_neg_prompt="[${context_neg_prompt}]"
  region1='[160, 76, 505, 350]'

  region2_prompt='a <catA1> <catA2> in galaxy, starwar background'
  region2_neg_prompt="[${context_neg_prompt}]"
  region2='[162, 370, 500, 685]'

  region3_prompt='a <dogA1> <dogA2> in galaxy, starwar background'
  region3_neg_prompt="[${context_neg_prompt}]"
  region3='[134, 666, 512, 1005]'

  prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}|${region3_prompt}-*-${region3_neg_prompt}-*-${region3}"

  CUDA_VISIBLE_DEVICES="0" python regionally_controlable_sampling.py \
    --pretrained_model=${fused_model} \
    --sketch_adaptor_weight=${sketch_adaptor_weight}\
    --sketch_condition=${sketch_condition} \
    --keypose_adaptor_weight=${keypose_adaptor_weight}\
    --keypose_condition=${keypose_condition} \
    --save_dir="results/multi-concept/orthogonal_animal_galaxy" \
    --prompt="${context_prompt}" \
    --negative_prompt="${context_neg_prompt}" \
    --prompt_rewrite="${prompt_rewrite}" \
    --suffix="baseline" \
    --seed=3
  
fi


if [ ${sample_dog} -eq 1 ]
then
  fused_model="experiments/composed_edlora/chilloutmix/animal/combined_model_base"

  keypose_condition=''
  keypose_adaptor_weight=1.0

  sketch_condition='datasets/validation_spatial_condition/multi-objects/dogA_catA_dogB.jpg'
  sketch_adaptor_weight=1.0

  context_prompt='<dogB1> <dogB2> and <catA1> <catA2> and <dogA1> <dogA2> in front of Mount Fuji'
  context_neg_prompt='dark, low quality, low resolution'

  region1_prompt='a <dogB1> <dogB2> in front of Mount Fuji'
  region1_neg_prompt="[${context_neg_prompt}]"
  region1='[160, 76, 505, 350]'

  region2_prompt='a <catA1> <catA2> in front of Mount Fuji'
  region2_neg_prompt="[${context_neg_prompt}]"
  region2='[162, 370, 500, 685]'

  region3_prompt='a <dogA1> <dogA2> in front of Mount Fuji'
  region3_neg_prompt="[${context_neg_prompt}]"
  region3='[134, 666, 512, 1005]'

  prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}|${region3_prompt}-*-${region3_neg_prompt}-*-${region3}"

  CUDA_VISIBLE_DEVICES="0" python regionally_controlable_sampling.py \
    --pretrained_model=${fused_model} \
    --sketch_adaptor_weight=${sketch_adaptor_weight}\
    --sketch_condition=${sketch_condition} \
    --keypose_adaptor_weight=${keypose_adaptor_weight}\
    --keypose_condition=${keypose_condition} \
    --save_dir="results/multi-concept/orthogonal_animal_fuji" \
    --prompt="${context_prompt}" \
    --negative_prompt="${context_neg_prompt}" \
    --prompt_rewrite="${prompt_rewrite}" \
    --suffix="baseline" \
    --seed=4
  
fi






