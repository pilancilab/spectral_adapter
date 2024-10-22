import clip
import torch
from PIL import Image
import PIL
from torchvision import transforms
import numpy as np
from argparse import ArgumentParser


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

    def img_to_img_similarity(self, src_images, generated_images):
        src_img_features = self.get_image_features(src_images)
        gen_img_features = self.get_image_features(generated_images)

        return (src_img_features @ gen_img_features.T).mean()

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
    parser = ArgumentParser()
    parser.add_argument("--exp_name", type=str, default='')
    parser.add_argument("--mode", type=str, default='validation')
    args = parser.parse_args()

    sources = []
    for index in range(1,7):
        sources.append(f"sources/{index}.jpg")
    
    samples = []
    for index in range(1,9):
        if args.mode == 'validation':
            samples.append(f"../Mix-of-Show/experiments/{args.exp_name}/visualization/PromptDataset/Iters-latest_Alpha-1.0/a_<vase1>_<vase2>_on_a_table---G_7.5_S_50---{index}---Iters-latest_Alpha-1.0.png")
        elif args.mode == 'test':
            samples.append(f"../Mix-of-Show/results/{args.exp_name}/visualization/PromptDataset/validation_edlora_1.0/a_<vase1>_<vase2>_on_a_table---G_7.5_S_50---{index}---validation_edlora_1.0.png")

    
    sources = [Image.open(i) for i in sources]
    sources = [i.convert("RGB") if not i.mode == "RGB" else i for i in sources]
    sources = [np.array(i).astype(np.uint8) for i in sources]
    sources = [Image.fromarray(i) for i in sources]
    sources = [i.resize((1080,1080), resample=PIL.Image.BICUBIC) for i in sources]
    sources = [np.array(i).astype(np.uint8) for i in sources]
    sources = [(i / 127.5 - 1.0).astype(np.float32) for i in sources]
    sources = [torch.from_numpy(i).permute(2, 0, 1) for i in sources]
    sources = torch.stack(sources, axis=0)

    samples = [Image.open(i) for i in samples]
    samples = [i.convert("RGB") if not i.mode == "RGB" else i for i in samples]
    samples = [np.array(i).astype(np.uint8) for i in samples]
    samples = [(i / 127.5 - 1.0).astype(np.float32) for i in samples]
    samples = [torch.from_numpy(i).permute(2, 0, 1) for i in samples]
    samples = torch.stack(samples, axis=0)



    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    clip_evaluator = CLIPEvaluator(device) 
    print(f'alignment score: {clip_evaluator.img_to_img_similarity(sources,samples)}')

