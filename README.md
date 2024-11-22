# FLAME


> **FLAME: Frozen Large Language Models Enable Data-Efficient Language-Image Pre-training** <br>
<a>Anjia Cao</a>,</span> <a>Xing Wei</a>,</span> <a>Zhiheng Ma</a><br>
[Paper](https://arxiv.org/abs/2411.11927) | [Data]()


## 📰 News
- [2024/11/18] [Paper](https://arxiv.org/abs/2411.11927) on arXiv.

## 💡 Highlights
- 🔥 Leveraging frozen LLMs to **naturally process long text inputs**.
- 🔥 Generalizing from monolingual training to **multilingual** evaluation.
- 🔥 Strong improvement on long/short-context image-text retrieval, image classification, and multilingual scenarios.

<img src="figures\long_context_1.png" style="vertical-align: -10px; display: block; margin-left: auto; margin-right: auto;" height="130px" width="290px">
<img src="figures\long_context_2.png" style="vertical-align: -10px; display: block; margin-left: auto; margin-right: auto;" height="130px" width="290px">
<img src="figures\multilingual_t2i_radar.png" style="vertical-align: -10px; display: block; margin-left: auto; margin-right: auto;" height="383px" width="331px">

## 📅 TODO Roadmap

- [ ] Release training code and data.
- [ ] Release evaluation code.
- [ ] Release pre-trained checkpoints.

## 🛠️ Get Started
#### Setup
```
git clone https://github.com/MIV-XJTU/FLAME.git
cd FLAME
conda create -n flame python=3.10 -y
conda activate flame
make install
make install-training
make install-test
```

#### Evaluation
Evaluate long-context retrieval.
```
TODO
```

Evaluate CLIP Benchmark (e.g. zero-shot classification, linear-probe classfication).
```
TODO
```

#### Training
TODO

## 🔐 Pretrained Checkpoints
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="center">Dataset</th>
<th valign="center">Model</th>
<th valign="center">Checkpoints</th>

<!-- TABLE BODY -->
<tr>
<td align="center">CC3M</td>
<td align="center">ViT-B/16</td>
<td align="center">TODO</td>
</tr>
<tr>
<td align="center">CC3M</td>
<td align="center">ViT-L/14</td>
<td align="center">TODO</td>
</tr>
<tr>
<td align="center">YFCC15M</td>
<td align="center">ViT-B/16</td>
<td align="center">TODO</td>
</tr>
<tr>
<td align="center">YFCC15M</td>
<td align="center">ViT-L/14</td>
<td align="center">TODO</td>
</tr>
</tbody></table>

## 📖 Citation
If you find our work helpful for your research, please consider giving a star and citation.
```bibtex
@article{cao2024flame,
  title={Flame: Frozen large language models enable data-efficient language-image pre-training},
  author={Cao, Anjia and Wei, Xing and Ma, Zhiheng},
  journal={arXiv preprint arXiv:2411.11927},
  year={2024}
}
```

### 🫡 Acknowledgements
This project is based on [open_clip](https://github.com/mlfoundations/open_clip), and thanks for the nice work! 
We also thank [CLIP_benchmark](https://github.com/LAION-AI/CLIP_benchmark), [DreamLIP](https://github.com/zyf0619sjtu/DreamLIP), [Long-CLIP](https://github.com/beichenzbc/Long-CLIP), and [PromptEOL](https://github.com/kongds/scaling_sentemb) for their codes.