## Download the Original Data & Our Recaptioned Data

[Download Link for Our Recaptioned CC3M](https://huggingface.co/datasets/caj/FLAME-ReCap-CC3M-MiniCPM-Llama3-V-2_5)

[Download Link for Our Recaptioned YFCC15M](https://huggingface.co/datasets/caj/FLAME-ReCap-YFCC15M-MiniCPM-Llama3-V-2_5)

## Embedding
Specify a slice of the webdataset for embedding.

e.g. for part of CC3M:

```
cd src

CUDA_VISIBLE_DEVICES=0  nohup python -u training/embed_cc3m_mask.py \
    --model FLAME-Mistral-Nemo-ViT-B-16 \
    --batch-size 32 \
    --workers 4 \
    --train-data '/data1/huggingface/hub/datasets--pixparse--cc3m-wds/snapshots/46f3d69f840e59d77d52e8decfe5baec97e94c7f/cc3m-train-{0000..0071}.tar' \
    --train-num-samples 363312 \
    > embed_cc3m_minicpm_shard_0000_0071.log 2>&1 &
```

## Training

```
cd src

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup   torchrun  --nproc_per_node 8 -m --master_port 12345 training.main \
    --train-data '/data1/huggingface/hub/datasets--pixparse--cc3m-wds/snapshots/46f3d69f840e59d77d52e8decfe5baec97e94c7f/cc3m-train-{0000..0575}.tar' \
    --imagenet-val /data1/imagenet/val/ \
    --zeroshot-frequency 2 \
    --batch-size 384 \
    --accum-freq 1 \
    --lr 5e-4 \
    --beta1 0.9 \
    --beta2 0.95 \
    --eps 1e-8 \
    --wd 0.5 \
    --warmup 2000 \
    --aug-cfg scale='(0.9, 1.0)' \
    --epochs 32 \
    --workers 4 \
    --model FLAME-Mistral-Nemo-ViT-B-16 \
    --mp-sigmoid \
    --precision 'amp' \
    --ddp-static-graph \
    --grad-checkpointing \
    --lock-text \
    --lock-text-freeze-layer-norm \
    --train-num-samples 2905954 \
    --log-every-n-steps 32 \
    --save-frequency 2 \
    --report-to wandb   >   FLAME-Mistral-Nemo-ViT-B-16.log 2>&1 &
```