import sys
import re
import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from open_clip import create_model_and_transforms
from training.params import parse_args
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F


def main(args):
    args = parse_args(args)
    device = 'cpu'
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

    image = Image.open('./tmp/image (10).jpg')
    transform = preprocess_val
    image = transform(image).unsqueeze(0).half()

    # model.visual.output_tokens = True
    model = model.half()
    local_image_tokens = model.visual.get_features(image)
    local_image_tokens = model.mm(local_image_tokens)
    _, image_features = model.visual._global_pool(local_image_tokens)

    bs = image_features.shape[0]
    embed_dim = image_features.shape[-1]
    patch_features = image_features.reshape(bs, 14, 14, embed_dim)
    image_features = F.avg_pool2d(patch_features.permute(0, 3, 1, 2), kernel_size=(2, 2), stride=(2, 2)).permute(0, 2, 3, 1).reshape(bs, 49, embed_dim)
    # image_features = F.avg_pool2d(patch_features.permute(0, 3, 1, 2), kernel_size=(4, 4), stride=(4, 4), padding=1).permute(0, 2, 3, 1).reshape(bs, 16, embed_dim)
    llm = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-Nemo-Instruct-2407', torch_dtype=torch.float16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-Nemo-Instruct-2407', trust_remote_code=True)

    logits_local = llm.lm_head(image_features)
    logits_local = logits_local.float()

    sorted_tensor, sorted_indices = torch.sort(logits_local, dim=-1, descending=True)
    output_ids_local = sorted_indices[:, :, 49]

    output_ids_local = output_ids_local.squeeze().tolist()

    words = ''
    for i in range(len(output_ids_local)):
        output_local = tokenizer.decode([output_ids_local[i]])
        print(output_local)
        words += f'{output_local}+'
    print(len(output_ids_local))
    print(words)


if __name__ == "__main__":
    main(sys.argv[1:])