# EVA: An Open-Domain Chinese Dialogue System with Large-Scale Generative Pre-Training

## Introduction

EVA is an open-domain Chinese pre-trained model, which contains the largest Chinese dialogue model with 2.8B parameters and is pre-trained on WDC-Dialogue, including 1.4B Chinese dialogue data from different domains. Our paper, data, codes and model parameters will be released soon.

## Dataset

We construct a dataset named **WDC-Dialogue** from Chinese social media to train EVA. Specifically, conversations from various sources are gathered and a rigorous data cleaning pipeline is designed to enforce the quality of WDC-Dialogue. We mainly focus on three categories of textual interaction data, i.e., **repost** on social media, **comment** / **reply** on various online forums and online **question and answer (Q\&A)** exchanges. Each round of these textual interactions yields a dialogue session via well-designed parsing rules. The following table shows a statistics of the filtered WDC-Dialogue dataset and other Chinese dialogue datasets.

<img src="fig/dataset.png" style="zoom:60%;" />

## Model

**EVA** is a Transformer-based dialogue model with a bi-directional encoder and a uni-directional decoder. We present the EVA's model details and a comparison with previous large-scale Chinese pre-trained dialogue models in the following table.

<img src="fig/model.png" style="zoom:60%;" />

## Experiment

We compare EVA with Chinese pre-trained models including [CDial-GPT](https://github.com/thu-coai/CDial-GPT) and [CPM](https://github.com/TsinghuaAI/CPM). Results in the automatic evaluation including uni-gram F1, ROUGE-L, BLEU-4 and distinct n-grams are shown as follows:

<img src="fig/auto_eval.png" style="zoom:50%;" />

We also present an example of multi-turn generation results in the interactive human evaluation:

<img src="fig/example.png" style="zoom:70%;" />

## Citation

```
@article{coai2021eva,
  title={EVA: An Open-Domain Chinese Dialogue System with Large-Scale Generative Pre-Training},
  author={Zhou, Hao and Ke, Pei and Zhang, Zheng and Gu, Yuxian and Zheng, Yinhe and Zheng, Chujie and Wang, Yida and Wu, Chen Henry and Sun, Hao and Yang, Xiaocong and Wen, Bosi and Zhu, Xiaoyan and Huang, Minlie and Tang, Jie},
  year={2021}
}
```
