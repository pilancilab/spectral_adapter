# Adapter Fusion Experiments for Spectral Adapter


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
 cd adapter_fusion
 conda create -n adapter_fusion python=3.10.13
 conda activate adapter_fusion
 pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
 pip install -r requirements.txt
 ```

## Preparation: Download Pretrained Models
```
 cd spectral/experiments/pretrained_models
 git-lfs clone https://huggingface.co/windwhinny/chilloutmix.git
 ```

## Our Method
### Step 1: Train single-concept spectral adapter
```
 cd spectral
 CUDA_VISIBLE_DEVICES=0  accelerate launch train_edlora.py -opt options/train/EDLoRA/animal/dogA.yml // adapter for dogA
 ```
replace <code>dogA.yml</code> with <code>dogB.yml,catA.yml</code> for other concepts.
### Step 2: Adapter fusion
```
sh spectral_fuse.sh
```
### Step 3: Image sampling
```
sh spectral_regionally_sample.sh 
```
results stored in <code>spectral/results/</code>.
## Baselines
### FedAvg:
```
 cd fedavg_gradient/experiments/pretrained_models
 git-lfs clone https://huggingface.co/windwhinny/chilloutmix.git
 cd ../..
 CUDA_VISIBLE_DEVICES=0  accelerate launch train_edlora.py -opt options/train/EDLoRA/animal/dogA.yml
 CUDA_VISIBLE_DEVICES=0  accelerate launch train_edlora.py -opt options/train/EDLoRA/animal/dogB.yml
 CUDA_VISIBLE_DEVICES=0  accelerate launch train_edlora.py -opt options/train/EDLoRA/animal/catA.yml
 sh fedavg_fuse.sh
 sh fedavg_regionally_sample.sh
 ```
results stored in <code>fedavg_gradient/results/</code>.
### Gradient Fusion:
can ignore pretrained model downloading and adapter training steps if already run FedAvg training.
```
 cd fedavg_gradient/experiments/pretrained_models
 git-lfs clone https://huggingface.co/windwhinny/chilloutmix.git
 cd ../..
 CUDA_VISIBLE_DEVICES=0  accelerate launch train_edlora.py -opt options/train/EDLoRA/animal/dogA.yml
 CUDA_VISIBLE_DEVICES=0  accelerate launch train_edlora.py -opt options/train/EDLoRA/animal/dogB.yml
 CUDA_VISIBLE_DEVICES=0  accelerate launch train_edlora.py -opt options/train/EDLoRA/animal/catA.yml
 sh gradient_fuse.sh
 sh gradient_regionally_sample.sh
 ```
results stored in <code>fedavg_gradient/results/</code>.
### Orthogonal Adaptation:
```
 cd orthogonal_adaptation/experiments/pretrained_models
 git-lfs clone https://huggingface.co/windwhinny/chilloutmix.git
 cd ../..
 CUDA_VISIBLE_DEVICES=0  accelerate launch train_edlora.py -opt options/train/EDLoRA/animal/dogA.yml
 CUDA_VISIBLE_DEVICES=0  accelerate launch train_edlora.py -opt options/train/EDLoRA/animal/dogB.yml
 CUDA_VISIBLE_DEVICES=0  accelerate launch train_edlora.py -opt options/train/EDLoRA/animal/catA.yml
 sh fuse.sh
 sh regionally_sample.sh
 ```
results stored in <code>orthogonal_adaptation/results/</code>.
## Reproducibility Shortcuts
for better reproducibility, we store the single-concept adapters we used for generating Figure 5. Therefore, one can directly fuse them for reproducing our results. After downloading pretrained models in each directory,
```
cd spectral
sh spectral_fuse.sh
sh fedavg_regionally_sample.sh
cd ../fedavg_gradient
sh fedavg_fuse.sh
sh fedavg_regionally_sample.sh
sh gradient_fuse.sh
sh gradient_regionally_sample.sh
cd ../orthogonal_adaptation
sh fuse.sh
sh regionally_sample.sh
```
check results in <code>spectral/results, fedavg_gradient/results, orthogonal_adaptation/results</code> respectively. We also provide detailed logs for each model trained and each picture generated (attached to each file) for guidance.

## Alignment Score Computation
To compute the image & text alignment score, after generating all figures and store them in <code>fedavg_gradient/results, fedavg_gradient/results, orthogonal_adaptation/results</code>, run
```
cd alignment_score
python image_alignment.py
```

