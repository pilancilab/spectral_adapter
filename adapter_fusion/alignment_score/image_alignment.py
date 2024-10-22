import clip
import torch
from PIL import Image
import PIL
from torchvision import transforms
import numpy as np
import sys 
import os


class CLIPEvaluator(object):
    def __init__(self, device, clip_model='ViT-B/32') -> None:
        self.device = device
        self.model, clip_preprocess = clip.load(clip_model, device=self.device)

        self.clip_preprocess = clip_preprocess
        
        self.preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] + # Un-normalize from [-1.0, 1.0] (generator output) to [0, 1].
                                              clip_preprocess.transforms[:2] +                                      # to match CLIP input scale assumptions
                                              clip_preprocess.transforms[4:])                                       # + skip convert PIL to tensor

    def tokenize(self, strings: list):
        return clip.tokenize(strings).to(self.device)

    @torch.no_grad()
    def encode_text(self, tokens: list) -> torch.Tensor:
        return self.model.encode_text(tokens)

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess(images).to(self.device)
        return self.model.encode_image(images)

    def get_text_features(self, text: str, norm: bool = True) -> torch.Tensor:

        tokens = clip.tokenize(text).to(self.device)

        text_features = self.encode_text(tokens).detach()

        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def get_image_features(self, img: torch.Tensor, norm: bool = True) -> torch.Tensor:
        image_features = self.encode_images(img)
        
        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features
    
    def get_image_features_crop(self, img: torch.Tensor, norm: bool = True) -> torch.Tensor:
        image_features = self.encode_images(img)
        
        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features

    def img_to_img_similarity(self, src_images, generated_images):

        sim_sum = 0
        src_images_cat = src_images[0:5,:]
        src_img_features = self.get_image_features(src_images_cat)
        gen_img_features = self.get_image_features_crop(generated_images[:,:,256:490,420:630])
        sim_sum += (src_img_features @ gen_img_features.T) .mean()
        src_images_dogA = src_images[5:10,:]
        src_img_features = self.get_image_features(src_images_dogA)
        gen_img_features = self.get_image_features_crop(generated_images[:,:,200:512,680:980])
        sim_sum += (src_img_features @ gen_img_features.T).mean()
        src_images_dogB = src_images[10:15,:]
        src_img_features = self.get_image_features(src_images_dogB)
        gen_img_features = self.get_image_features_crop(generated_images[:,:,220:490,100:300])
        sim_sum += (src_img_features @ gen_img_features.T).mean()

        return sim_sum/3
    
    def txt_to_img_similarity(self, text, generated_images):
        text_features    = self.get_text_features(text)
        gen_img_features = self.get_image_features(generated_images)

        return (text_features @ gen_img_features.T).mean()


class ImageDirEvaluator(CLIPEvaluator):
    def __init__(self, device, clip_model='ViT-B/32') -> None:
        super().__init__(device, clip_model)

    def evaluate(self, gen_samples, src_images, target_text):

        sim_samples_to_img  = self.img_to_img_similarity(src_images, gen_samples)
        sim_samples_to_text = self.txt_to_img_similarity(target_text.replace("*", ""), gen_samples)

        return sim_samples_to_img, sim_samples_to_text
    

if __name__ == '__main__':
    cat_1 = "animal_reference/cat/cat1.jpg"
    cat_2 = "animal_reference/cat/cat2.jpg"
    cat_3 = "animal_reference/cat/cat3.jpg"
    cat_4 = "animal_reference/cat/cat4.jpg"
    cat_5 = "animal_reference/cat/cat5.jpg"

    doga_1 = "animal_reference/dogA/dog1.jpg"
    doga_2 = "animal_reference/dogA/dog2.jpg"
    doga_3 = "animal_reference/dogA/dog3.jpg"
    doga_4 = "animal_reference/dogA/dog4.jpg"
    doga_5 = "animal_reference/dogA/dog5.jpg"

    dogb_1 = "animal_reference/dogB/dog1.jpg"
    dogb_2 = "animal_reference/dogB/dog2.jpg"
    dogb_3 = "animal_reference/dogB/dog3.jpg"
    dogb_4 = "animal_reference/dogB/dog4.jpg"
    dogb_5 = "animal_reference/dogB/dog5.jpg"

    folder = {'spectral':'spectral',
              'fedavg':'fedavg_gradient',
              'gradient':'fedavg_gradient',
              'orthogonal':'orthogonal_adaptation'}
    
    txt = {'fuji':"a dog and a cat and a dog in front of Mount Fuji",
           'playground':"a dog and a cat and a dog on a playground, in school",
           'galaxy':"a dog and a cat and a dog in galaxy, starwar background"}
    


    for scene in ['fuji','playground','galaxy']:
        for method in ['spectral','fedavg','gradient','orthogonal']:
            sample_1 = f"../{folder[method]}/results/multi-concept/{method}_animal_{scene}/pic.png"

            sources = [cat_1,cat_2,cat_3,cat_4,cat_5,doga_1,doga_2,doga_3,doga_4,doga_5,dogb_1,dogb_2,dogb_3,dogb_4,dogb_5]
            sources = [Image.open(i) for i in sources]
            sources = [i.convert("RGB") if not i.mode == "RGB" else i for i in sources]
            sources = [np.array(i).astype(np.uint8) for i in sources]
            sources = [Image.fromarray(i) for i in sources]
            sources = [i.resize((1080,1080), resample=PIL.Image.BICUBIC) for i in sources]
            sources = [np.array(i).astype(np.uint8) for i in sources]
            sources = [(i / 127.5 - 1.0).astype(np.float32) for i in sources]
            sources = [torch.from_numpy(i).permute(2, 0, 1) for i in sources]
            sources = torch.stack(sources, axis=0)

            samples = [sample_1] 
            samples = [Image.open(i) for i in samples]
            samples = [i.convert("RGB") if not i.mode == "RGB" else i for i in samples]
            samples = [np.array(i).astype(np.uint8) for i in samples]
            samples = [(i / 127.5 - 1.0).astype(np.float32) for i in samples]
            samples = [torch.from_numpy(i).permute(2, 0, 1) for i in samples]
            samples = torch.stack(samples, axis=0)

            device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
            clip_evaluator = CLIPEvaluator(device) 
            print(f'scene: {scene}   method: {method}')
            img_score = clip_evaluator.img_to_img_similarity(sources,samples)
            txt_score = clip_evaluator.txt_to_img_similarity(txt[scene], samples)
            print(f'img score: {img_score}  txt score: {txt_score}   avg: {(img_score+txt_score)/2}')
            print('\n')

