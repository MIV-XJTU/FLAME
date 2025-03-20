# CVPR 2025 | FLAME


> **FLAME: Frozen Large Language Models Enable Data-Efficient Language-Image Pre-training** <br>
<a>Anjia Cao</a>,</span> <a>Xing Wei</a>,</span> <a>Zhiheng Ma</a>

 
## 📰 News
- [2025/03/09] Release training codes and [scripts](./Training.md).
- [2025/03/08] [Recaptioned CC3M](https://huggingface.co/datasets/caj/FLAME-ReCap-CC3M-MiniCPM-Llama3-V-2_5) and [Recaptioned YFCC15M](https://huggingface.co/datasets/caj/FLAME-ReCap-YFCC15M-MiniCPM-Llama3-V-2_5) on Hugging Face.
- [2025/02/27] Accepted by CVPR 2025.
- [2024/11/28] [Model](https://huggingface.co/caj/FLAME-Mistral-Nemo-ViT-B-16) on Hugging Face.
- [2024/11/28] Release evaluation codes and [scripts](./Evaluation.md).
- [2024/11/18] [Paper](https://arxiv.org/abs/2411.11927) on arXiv.

## 💡 Highlights
- 🔥 Leveraging frozen LLMs to **naturally process long text inputs**.
- 🔥 Generalizing from monolingual training to **multilingual** evaluation.
- 🔥 Strong improvement on long/short-context image-text retrieval, image classification, and multilingual scenarios.

<img src="figures\long_context.png" style="vertical-align: -10px; display: block; margin-left: auto; margin-right: auto;" height="200px" width="630px">
<img src="figures\multilingual.png" style="vertical-align: -10px; display: block; margin-left: auto; margin-right: auto;" height="400px" width="670px">

## 📅 TODO Roadmap

- [x] Release training code and data.
- [x] Release evaluation code.
- [x] Release pre-trained checkpoints.

## 🛠️ Get Started
### Setup
```
git clone https://github.com/MIV-XJTU/FLAME.git
cd FLAME
conda create -n flame python=3.10 -y
conda activate flame
make install
make install-training
make install-test
```

### Training
See [Training.md](./Training.md).

### Evaluation
See [Evaluation.md](./Evaluation.md).

## 📁 Datasets
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="center">Dataset</th>
<th valign="center">Link</th>
<!-- TABLE BODY -->
<tr>
<td align="center">CC3M-ReCap</td>
<td align="center"><a href="https://huggingface.co/datasets/caj/FLAME-ReCap-CC3M-MiniCPM-Llama3-V-2_5">Hugging Face</a></td>
</tr>
<tr>
<td align="center">YFCC15M-ReCap</td>
<td align="center"><a href="https://huggingface.co/datasets/caj/FLAME-ReCap-YFCC15M-MiniCPM-Llama3-V-2_5">Hugging Face</a></td>
</tr>
</tbody></table>

## 🔐 Pre-trained Checkpoints
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="center">Dataset</th>
<th valign="center">Model</th>
<th valign="center">Link</th>

<!-- TABLE BODY -->
<tr>
<td align="center">CC3M</td>
<td align="center">ViT-B/16-Mistral-Nemo</td>
<td align="center"><a href="https://huggingface.co/caj/FLAME-Mistral-Nemo-ViT-B-16">Hugging Face</a></td>
</tr>
<tr>
<td align="center">YFCC15M</td>
<td align="center">ViT-B/16-Mistral-Nemo</td>
<td align="center"><a href="https://huggingface.co/caj/FLAME-Mistral-Nemo-ViT-B-16">Hugging Face</a></td>
</tr>
<tr>
<td align="center">Merged-2B</td>
<td align="center">ViT-L/14-Qwen2.5</td>
<td align="center">TODO</td>
</tr>
</tbody></table>

## 🛂 License

The project is under a standard Creative Common [CC-BY-4.0 License](./LICENSE).

## 📖 Citation
If you find our work helpful for your research, please consider giving a star and citation.
```bibtex
@article{cao2024flame,
  title={FLAME: Frozen Large Language Models Enable Data-Efficient Language-Image Pre-training},
  author={Cao, Anjia and Wei, Xing and Ma, Zhiheng},
  journal={arXiv preprint arXiv:2411.11927},
  year={2024}
}
```

## 🫡 Acknowledgements
This project is based on [open_clip](https://github.com/mlfoundations/open_clip), and thanks for the nice work! 
We also thank [CLIP_benchmark](https://github.com/LAION-AI/CLIP_benchmark), [DreamLIP](https://github.com/zyf0619sjtu/DreamLIP), [Long-CLIP](https://github.com/beichenzbc/Long-CLIP), [PromptEOL](https://github.com/kongds/scaling_sentemb), and [MiniCPM-V](https://github.com/OpenBMB/MiniCPM-o) for their codes.
