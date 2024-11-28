## Evaluation

* **For long text inputs, inference with multiple prompts.**
Set ```multi_template=True``` of ```encode_text()``` in [src/open_clip/model.py](./src/open_clip/model.py).
  * [ShareGPT4V](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/blob/main/share-captioner_coco_lcs_sam_1246k_1107.json), [Urban-1k](https://huggingface.co/datasets/BeichenZhang/Urban1k/blob/main/Urban1k.zip), [DCI](https://github.com/facebookresearch/DCI), and [DOCCI](https://google.github.io/docci/#downloads):
    ```
    cd src

    CUDA_VISIBLE_DEVICES=0 python -u training/eval_$dataset$.py \
    --model FLAME-ViT-B-16 \
    --pretrained $path_to_ckpt$
    ```
  * [MSCOCO](https://huggingface.co/datasets/clip-benchmark/wds_mscoco_captions) and [Flickr30k](https://huggingface.co/datasets/clip-benchmark/wds_flickr30k):
    ```
    cd $path_to_clip_benchmark$/benchmark

    CUDA_VISIBLE_DEVICES=0 clip_benchmark eval \
    --dataset wds/$dataset$ \
    --dataset_root $path_to_dataset$ \
    --task zeroshot_retrieval \
    --pretrained $path_to_ckpt$ \
    --model FLAME-ViT-B-16 \
    --output ./outputs/zs_retrieval/$dataset$.json \
    --batch_size 64 \
    --recall_k 1 5 10
    ```
  * [Winoground](https://huggingface.co/datasets/facebook/winoground) and [SugarCrepe](https://github.com/RAIVNLab/sugar-crepe):
    ```
    pip install datasets
    cd $path_to_clip_benchmark$/benchmark

    CUDA_VISIBLE_DEVICES=0 clip_benchmark eval \
    --dataset winoground \
    --pretrained $path_to_ckpt$ \
    --model FLAME-ViT-B-16 \
    --output ./outputs/compositionality/winoground.json 

    CUDA_VISIBLE_DEVICES=0 clip_benchmark eval \
    --dataset sugar_crepe/add_att sugar_crepe/add_obj sugar_crepe/replace_att sugar_crepe/replace_obj sugar_crepe/replace_rel sugar_crepe/swap_att sugar_crepe/swap_obj \
    --pretrained $path_to_ckpt$ \
    --model FLAME-ViT-B-16 \
    --output ./outputs/compositionality/{dataset}.json 
    ```
* **For short text inputs, inference with single prompt.**
Set ```multi_template=False``` of ```encode_text()``` in [src/open_clip/model.py](./src/open_clip/model.py).
  * [Crossmodal3600](https://google.github.io/crossmodal-3600):
    ```
    cd $path_to_clip_benchmark$/benchmark

    CUDA_VISIBLE_DEVICES=0 clip_benchmark eval \
    --dataset crossmodal3600 \
    --task zeroshot_retrieval \
    --pretrained $path_to_ckpt$ \
    --model FLAME-ViT-B-16 \
    --output ./outputs/multilingual_retrieval/crossmodal_{language}.json \
    --batch_size 16 \
    --language ar bn cs da de el en es fa fi fil fr he hi hr hu id it ja ko mi nl no pl pt quz ro ru sv sw te th tr uk vi zh \
    --recall_k 1
    ```
  * Zero-shot image classficiation:
    ```
    cd $path_to_clip_benchmark$/benchmark

    CUDA_VISIBLE_DEVICES=0 clip_benchmark eval \
    --dataset wds/$dataset$ \
    --dataset_root $path_to_dataset$ \
    --task zeroshot_classification \
    --pretrained $path_to_ckpt$ \
    --model FLAME-ViT-B-16 \
    --output ./outputs/zs_classification/$dataset$.json \
    --batch_size 64
    ```
  * Linear-probe classification:
    Set ```visual_only=True``` of ```encode_image()``` in [src/open_clip/model.py](./src/open_clip/model.py).
    ```
    cd $path_to_clip_benchmark$/benchmark

    CUDA_VISIBLE_DEVICES=0 clip_benchmark eval \
    --dataset wds/$dataset$ \
    --dataset_root $path_to_dataset$ \
    --task linear_probe \
    --pretrained $path_to_ckpt$ \
    --model FLAME-ViT-B-16 \
    --output ./outputs/lp_classification/$dataset$.json \
    --batch_size 64 \
    --fewshot_lr 0.1 \
    --fewshot_epochs 20 \
    --batch_size 512 \
    --train_split train \
    --test_split test
    ```
  * Multilingual [ImageNet1k](https://www.image-net.org) classification:
    ```
    cd $path_to_clip_benchmark$/benchmark

    CUDA_VISIBLE_DEVICES=0 clip_benchmark eval \
    --dataset imagenet1k \
    --dataset_root $path_to_imagenet1k$ \
    --task zeroshot_classification \
    --pretrained $path_to_ckpt$ \
    --model FLAME-ViT-B-16 \
    --output ./outputs/multilingual_classification/imagenet1k_{language}.json \
    --batch_size 64 \
    --language ar en jp it cn 
    ```