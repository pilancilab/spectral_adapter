# Adapter Efficiency Experiments for Spectral Adapter


**Spectral Adapter: Fine-Tuning in Spectral Space** <br>
*Fangzhao Zhang, Mert Pilanci* <br>
Paper: [https://arxiv.org/abs/2405.13952](https://arxiv.org/abs/2405.13952) <br>

This repository is for reproducing Figure 5 result:
<p>
<img src="figures/animal_img2-01.png" width="800" >
</p>

## Quickstart
Clone the repo and run the following command
 ```
 cd adapter_efficiency
 conda create -n adapter_efficiency python=3.10.13
 conda activate adapter_efficiency
 pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
 pip install -r requirements.txt
 ```

## Preparation: Download Pretrained Models
```
 cd mix_spectral/Mix-of-Show/experiments/pretrained_models
 git-lfs clone https://huggingface.co/windwhinny/chilloutmix.git
 ```

## Our Method
### Train single-concept spectral adapter
```
 cd mix_spectral/Mix-of-Show
 CUDA_VISIBLE_DEVICES=0  accelerate launch train_edlora.py -opt options/train/EDLoRA/object/vase.yml
 ```
results stored in <code>mix_spectral/Mix-of-Show/experiments/exp_name/visualization/PromptDataset/</code>. <code>exp_name</code> can be found/modified in <code>Mix-of-Show/options/train/EDLoRA/object/vase.yml</code> 
### Compute alignment score
```
 cd mix_spectral/alignment_score
 python image_alignment.py --exp_name=exp_name
 ```

## Baselines
To run our baseline methods, 
### LoRA:
```
 cd mix_lora/Mix-of-Show
 CUDA_VISIBLE_DEVICES=0  accelerate launch train_edlora.py -opt options/train/EDLoRA/object/vase.yml
``` 
### VeRA:
```
 cd mix_vera/Mix-of-Show
 CUDA_VISIBLE_DEVICES=0  accelerate launch train_edlora.py -opt options/train/EDLoRA/object/vase.yml
``` 
### LiDB:
```
 cd mix_lidb/Mix-of-Show
 CUDA_VISIBLE_DEVICES=0  accelerate launch train_edlora.py -opt options/train/EDLoRA/object/vase.yml
``` 
### SVDiff:
```
 cd mix_svdiff/Mix-of-Show
 CUDA_VISIBLE_DEVICES=0  accelerate launch train_edlora.py -opt options/train/EDLoRA/object/vase.yml
``` 
### OFT
```
 cd mix_oft/Mix-of-Show
 CUDA_VISIBLE_DEVICES=0  accelerate launch train_edlora.py -opt options/train/EDLoRA/object/vase.yml
```
for convenience, each subfolder contains an alignement computation file, just run
```
 de ../alignment_score
 python image_alignment.py --exp_name=exp_name
```

## Reproducibility Tips
For better reproducibility, we provide detail logs to all models trained and images generated (attached to each file). Due to github file size limitation, we remove original model file. We report average score over three random trials generations for our method.

See table below for training hyperparameters we used for each method, the corresponding ranks and lrs can be edited in <code>Mix-of-Show/options/train/EDLoRA/object/vase.yml</code>  in each subfolder.

| Method  | ranks | text_encoder lr  | unet lr  | 
| ------------- | ------------- | ------------- |  ------------- |  
| LoRA  | r=1,2,3 | 1e-5 | 1e-4 |  
| VeRA | r=1  | 1e-3  | 1e-4 | 
| VeRA | r=1024,4096 | 5e-3  | 1e-4 | 
| OFT  | r=8,16,32,64 | 1e-5  | 1e-4 |  
| SVDiff  | n/a | 1e-3  | 1e-4 | 
| LiDB  |r=1,16,32 | 5e-4  | 1e-4 |  
| Spectral | r=2,40  | 1e-3 | 1e-2 | 
| Spectral | r=8  | 5e-4 | 5e-2 | 
| Spectral | r=24  | 1e-4 | 1e-2 | 
| Spectral | r=32  | 1e-4 | 5e-2 | 

We also print corresponding # trainable params as part of training progression logs.

## Acknowledgements
This repo is directly extended from [Mix-of-Show](https://github.com/TencentARC/Mix-of-Show/tree/main). 
