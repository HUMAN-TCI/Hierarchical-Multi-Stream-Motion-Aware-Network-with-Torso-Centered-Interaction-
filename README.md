# HUMAN-TCI: Hierarchical Multi-Stream Motion-Aware Network with Torso-Centered Interaction for Text-to-Motion Retrieval

## Overview
HUMAN-TCI is a text-to-motion retrieval framework that learns a joint embedding space between natural language descriptions and human motion sequences. The model integrates:

- **Text Encoders**: BERT-LSTM and CLIP (for rich semantic understanding)
- **Motion Encoder**: Hierarchical Multi-Stream GRU (capturing body-part level dynamics)
- **Interaction Mechanism**: Torso-Centered Interaction (TCI) for improved spatial-temporal alignment

The system retrieves the most relevant motion sequence given a text query (and vice versa).

---

## Features
- Hierarchical multi-stream architecture modeling upper-body, lower-body, and torso interactions
- Explicit torso-aware modeling with cross-part dependency learning
- Handles complex, multi-action, and compositional text descriptions
- Strong and efficient performance on KIT Motion-Language Dataset and HumanML3D

---

## Project Structure
```
├── train.py                # Training script
├── inference.py            # Retrieval / evaluation script
├── render.py               # Visualization script
├── models/                 # Model architectures (text + motion encoders)
├── utils/                  # Helper functions (metrics, preprocessing, etc.)
├── data/                   # Dataset directory
├── checkpoints/            # Saved model weights
├── outputs/                # Retrieved results and videos
└── README.md
```

---

## Installation

### Requirements
- Python 3.8+
- PyTorch (tested on 1.12.0+cu102)
- CUDA (optional, for GPU acceleration)

### Install dependencies
```bash
pip install -r requirements.txt
```

---

## Dataset Preparation

Prepare the dataset (e.g., KIT-ML or HumanML3D) in the following format: Download from authors work

```
data/
├── motions/
├── texts/
└── splits/
```

Ensure motion features and text annotations are properly aligned.

---

## Training

To train the model:

```bash
python train.py --config configs/train.yaml
```

Key options:
- `text_model`: bert-lstm / clip
- `motion_model`: multi-stream-gru
- `batch_size`, `learning_rate`, etc.

---

## Inference / Evaluation

To evaluate retrieval performance:

```bash
python inference.py --checkpoint checkpoints/model.pth --split test
```

Metrics reported:
- R@1, R@5, R@10
- Median Rank (MedR)
- Mean Rank (MeanR)
- NDCG - Spacy, Spice

---
## 🎥 Results

### Comparison with State-of-the-Art Methods on KIT-ML and HumanML3D

| Method | Pub' Year | KIT-ML R@1 ↑ | R@5 ↑ | R@10 ↑ | MedR ↓ | HumanML3D R@1 ↑ | R@5 ↑ | R@10 ↑ | MedR ↓ |
|--------|----------|--------------|------|--------|--------|------------------|------|--------|--------|
| T2M | CVPR'22 | 3.37 | 16.87 | 27.71 | 28 | 1.80 | 7.12 | 12.47 | 81 |
| MotionCLIP | ECCV'22 | 4.87 | 20.09 | 31.57 | 26 | 2.33 | 12.77 | 18.14 | 103 |
| TEMOS | ECCV'22 | **7.11** | 24.10 | 35.66 | 24 | 2.12 | 8.26 | 13.52 | 173 |
| MoT | SIGIR'23 | 6.23 | 23.92 | 37.15 | 20 | 2.61 | 10.66 | 17.79 | 60 |
| TMR | ICCV'23 | _**7.23**_ | _**28.31**_ | **40.12** | **17** | _**5.68**_ | **20.34** | **30.94** | _**28**_ |
| Messi-B | SIGIR'23 | 3.20 | 15.70 | 25.30 | 34 | 2.40 | 10.50 | 17.70 | 68 |
| DTL | MM'24 | 6.77 | 23.18 | 37.24 | 18 | 2.30 | 10.06 | 16.40 | 76 |
| **HUMAN-TCI (Ours)** | -- | 6.82 | **26.53** | _**41.26**_ | _**16**_ | **4.82** | _**22.10**_ | _**33.54**_ | **37** |

## 🎥 Qualitative Results (A- Skeleton Based)

<table>
<tr>
<td align="center">
<img src="render outputs/Skeleteon/128/0_M011988.npy.gif" width="260"/><br>
<sub>A person bows forward to their waist somewhat slowly</sub>
</td>

<td align="center">
<img src="render outputs/Skeleteon/113/0_005139.npy.gif" width="260"/><br>
<sub>A person is standing with arms out, then sits and rests hands on knees</sub>
</td>

<td align="center">
<img src="render outputs/Skeleteon/129/0_M010456.npy.gif" width="260"/><br>
<sub>A person is dancing the waltz counter-clockwise with the left arm out</sub>
</td>
</tr>

<tr>
<td align="center">
<img src="render outputs/Skeleteon/56/0_M001952.npy.gif" width="260"/><br>
<sub>A person walks in a counterclockwise circle</sub>
</td>

<td align="center">
<img src="render outputs/Skeleteon/70/0_001781.npy.gif" width="260"/><br>
<sub>A person does squats with raised hands, lifting them overhead when standing</sub>
</td>

<td align="center">
<img src="render outputs/Skeleteon/79/0_M007514.npy.gif" width="260"/><br>
<sub> The person is waving with their left arm.</sub>
</td>

<td></td>
</tr>
</table>

## 🎥 Qualitative Results (B- SMPL Based)

<table>
<tr>
<td align="center">
<img src="render outputs/SMPL Video/135/M001840_mesh.gif" width="260"/><br>
<sub>A man is standing and brings both hands to his face then steps out with left foot and performs a low kick.
</sub>
</td>

<td align="center">
<img src="render outputs/SMPL Video/155/M003897_mesh.gif" width="260"/><br>
<sub>A man gets on his knees and crawls from right to left, then stands up again.
</sub>
</td>

<td align="center">
<img src="render outputs/SMPL Video/203/M005433_mesh.gif" width="260"/><br>
<sub>A person backed up and sat down</sub>
</td>
</tr>

<tr>
<td align="center">
<img src="render outputs/SMPL Video/209/009577_mesh.gif" width="260"/><br>
<sub>A person puts his hands together in front of him then rests them on his side</sub>
</td>

<td align="center">
<img src="render outputs/SMPL Video/22/004965_mesh.gif" width="260"/><br>
<sub> A person walks up to something, picks it up, brings it back to where they were, and begins to make a washing motion with their hand.</sub>
</td>

<td align="center">
<img src="render outputs/SMPL Video/243/009041.gif" width="260"/><br>
<sub> A person standing up throws something forward from above their head, then throws something again forward from above their head with more force which makes them take one step forward with their right foot.</sub>
</td>

<td></td>
</tr>
</table>

## 🎥 Qualitative Results (Full Frame ( Starting-Ending )

<table>
<tr>
<td align="center">
<video src="render outputs/SMPL Video/135/M001840_mesh.gif" autoplay loop muted width="260"></video><br>
<sub>A man stands, brings both hands to his face, then steps out with his left foot and performs a low kick.</sub>
</td>

<td align="center">
<video src="render outputs/SMPL Video/155/M003897_mesh.gif" autoplay loop muted width="260"></video><br>
<sub>A man gets on his knees, crawls from right to left, and then stands up again.</sub>
</td>

<td align="center">
<video src="render outputs/SMPL Video/203/M005433_mesh.gif" autoplay loop muted width="260"></video><br>
<sub>A person steps backward and sits down.</sub>
</td>
</tr>

<tr>
<td align="center">
<video src="render outputs/SMPL Video/209/009577_mesh.gif" autoplay loop muted width="260"></video><br>
<sub>A person brings their hands together in front, then lowers them to their sides.</sub>
</td>

<td align="center">
<video src="render outputs/SMPL Video/22/004965_mesh.gif" autoplay loop muted width="260"></video><br>
<sub>A person walks forward, picks something up, returns, and performs a washing motion.</sub>
</td>

<td align="center">
<video src="render outputs/SMPL Video/243/009041.gif" autoplay loop muted width="260"></video><br>
<sub>A person throws forward twice, stepping forward with the right foot on the second throw.</sub>
</td>
</tr>
</table>

**Note:** More results are available in the `render outputs/` directory.

## Outputs

The system produces:
- Retrieval rankings
- Quantitative evaluation metrics
- Visualization videos of motion sequences

---

## Supplementary Material

The repository includes:
- Full training and inference code
- Pretrained model checkpoints (if provided)
- Visualization scripts
- Output videos demonstrating retrieval performance

---

## Citation

If you find this work useful, please cite:

```
@article{human_tci,
  title={HUMAN-TCI: Hierarchical Multi-Stream Motion-Aware Network with Torso-Centered Interaction for Text-to-Motion Retrieval},
  author={Anonymous Author(s) },
  year={2026}
}
```

---



## Acknowledgements

This work builds upon prior research in text-to-motion retrieval, multimodal learning, and human motion modeling.

Our implementation is largely based on the following works, and we sincerely thank the authors for making their codebases publicly available:

MDM: Human Motion Diffusion Model
https://github.com/GuyTevet/motion-diffusion-model

TMR: Text-to-Motion Retrieval Using Contrastive 3D Human Motion Synthesis
https://mathis.petrovich.fr/tmr/

Text-to-Motion Retrieval: Towards Joint Understanding of Human Motion Data and Natural Language.
https://github.com/mesnico/text-to-motion-retrieval

Their contributions have been instrumental in advancing research in this domain and have significantly supported the development of this project.

