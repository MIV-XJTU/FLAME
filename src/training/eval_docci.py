import sys
import re
import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from open_clip import create_model_and_transforms, get_tokenizer, SimpleTokenizer
from training.params import parse_args
# from training.docci import docci_val_dataset
from torchvision import transforms
from PIL import Image
from contextlib import suppress

import json
import cv2
import torch.utils.data as data
import os
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


datadocci_root = '/data1/datasets/docci/'
json_name = 'docci_descriptions.jsonlines'
image_root = '/data1/datasets/docci/images/'


class docci_val_dataset(data.Dataset):
    def __init__(self, transform=None, tokenizer=None):
        self.datadocci_root = datadocci_root
        self.json_name = json_name
        self.image_root = image_root
        # self.total_len = 1000
        # self.transform = transform
        # self.preprocess_txt = lambda text: tokenizer(text)[0]
        self.json_data = []
        with open(datadocci_root + json_name, 'r',encoding='utf8')as fp:
            for line in fp:
                json_data = json.loads(line)
                if json_data["split"] == 'test':
                    self.json_data.append(json_data)
        # _ , self.preprocess = clip.load("ViT-L/14")

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):
        uid = self.json_data[index]['example_id']
        caption = self.json_data[index]['description']
        caption = caption.replace("\n", " ")
        image_name = self.image_root + self.json_data[index]['image_file']
        try:
            image = Image.open(image_name)
            # image_tensor = self.transform(image)
        except Exception as e:
            print(f"err: {e}, file: {image_name}")
        # caption = self.preprocess_txt(caption)
        # return uid, image_tensor, caption
        return uid, image, caption


def main(args):
    args = parse_args(args)
    device = 'cuda'
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=args.force_patch_dropout,
        force_image_size=args.force_image_size,
        image_mean=args.image_mean,
        image_std=args.image_std,
        image_interpolation=args.image_interpolation,
        image_resize_mode=args.image_resize_mode,  # only effective for inference
        aug_cfg=args.aug_cfg,
        pretrained_image=args.pretrained_image,
        output_dict=True,
    )   
    checkpoint = torch.load(args.pretrained, map_location='cpu')
    sd = checkpoint["state_dict"]
    new_sd = {}
    for k, v in sd.items():
        name = k[7:]
        new_sd[name] = v
    model.load_state_dict(new_sd, strict=False)
    print(model)
    tokenizer = get_tokenizer(args.model)

    # model, _, preprocess = create_model_and_transforms('ViT-B-16', pretrained=args.pretrained)
    # model = model.to(device).eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
    # # tokenizer = get_tokenizer('ViT-B-16')
    # tokenizer = SimpleTokenizer()

    def _convert_image_to_rgb(image):
        return image.convert("RGB")

    preprocess = Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    dataset = docci_val_dataset(transform=preprocess, tokenizer=tokenizer)
    print(len(dataset))
    text_feature_list = []
    img_feature_list = []
    text_list_1 = []
    text_list_2 = []
    text_list = []
    correct = 0
    total = 0

    with torch.no_grad(), torch.cuda.amp.autocast():
        for i, (uid, image, caption) in enumerate(dataset):
            text_feature = model.text.tokenizer([caption], return_tensors='pt', max_length=model.text.max_length, padding='max_length', truncation=True).input_ids.to(device)
            text_feature = model.encode_text(text_feature, template_type='long', multi_template=True, normalize=True)
            # text_feature = tokenizer([caption]).to(device)
            # text_feature = model.encode_text(text_feature, normalize=True)
            text_feature_list.append(text_feature)

        text_embeds = torch.cat(text_feature_list, dim=0)
        text_embeds /= text_embeds.norm(dim=-1, keepdim=True)

        for i, (uid, image, caption) in enumerate(dataset):            
            image = preprocess(image).half().unsqueeze(0).to(device)
            img_feature = model.encode_image(image, normalize=True)
            img_feature_list.append(img_feature)
            
        image_embeds = torch.cat(img_feature_list, dim=0)
        image_embeds /= image_embeds.norm(dim=-1, keepdim=True)
        
        print("text 2 image")
        i = 0
        correct = 0
        total = 0
        for i in range(text_embeds.shape[0]):
            text = text_embeds[i]
            sim = text @ image_embeds.T
            sim = sim.squeeze()
            correct_i = torch.argmax(sim)

            if i==correct_i:
                correct = correct + 1
            total = total + 1
        print(total)
        print(correct)
        print(correct/total)
        
        print("image to text")
        i = 0
        correct = 0
        total = 0
        for i in range(image_embeds.shape[0]):
            img = image_embeds[i]
            sim = img @ text_embeds.T
            sim = sim.squeeze()
            correct_i = torch.argmax(sim)

            if i==correct_i:
                correct = correct + 1
            total = total + 1
        print(total)
        print(correct)
        print(correct/total)


if __name__ == "__main__":
    main(sys.argv[1:])