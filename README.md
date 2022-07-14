# README

本分支是 EVA 代码的 HuggingFace版本，原始版本请见 [main 分支](https://github.com/thu-coai/EVA/tree/main)。

## 1 模型下载

从 HuggingFace 模型仓库中下载，现在支持以下模型：
+ [EVA2.0-xLarge](https://huggingface.co/thu-coai/EVA2.0-xlarge)

## 2 环境安装

```[bash]
pip3 install -r requirements
```

## 3 使用示例

如 `src/example.py` 所示：

```[python]
from model import EVAModel, EVATokenizer

tokenizer = EVATokenizer.from_pretrained("/PATH-TO-EVA-CHECKPOINT/")
model = EVAModel.from_pretrained("/PATH-TO-EVA-CHECKPOINT/")
model = model.half().cuda()

input_str = "今天天气怎么样"

tokenize_out = tokenizer(input_str, "", return_tensors="pt", padding=True, truncation=True, max_length=512)
input_ids = tokenize_out.input_ids.cuda()

gen = model.generate(input_ids, do_sample=True, decoder_start_token_id=tokenizer.bos_token_id, top_p=0.9, max_length=32, use_cache=True)
print(tokenizer.decode(gen[0], skip_special_tokens=True))
```

## 4 生成测试

```[bash]
bash scripts/eva_inference_interactive.sh # 交互式生成
bash scripts/eva_inference_static.sh # 静态评测
```

**注意**：我们的[原始版本](https://github.com/thu-coai/EVA/tree/main)对 HuggingFace 封装的解码算法进行了一定的优化，因此生成性能会比此版本稍好一些。

## 5 引用

```
@article{coai2021eva,
  title={{EVA}: An Open-Domain Chinese Dialogue System with Large-Scale Generative Pre-Training},
  author={Zhou, Hao and Ke, Pei and Zhang, Zheng and Gu, Yuxian and Zheng, Yinhe and Zheng, Chujie and Wang, Yida and Wu, Chen Henry and Sun, Hao and Yang, Xiaocong and Wen, Bosi and Zhu, Xiaoyan and Huang, Minlie and Tang, Jie},
  journal={arXiv preprint arXiv:2108.01547},
  year={2021}
}
@article{coai2022eva2,
  title={{EVA2.0}: Investigating Open-Domain Chinese Dialogue Systems with Large-Scale Pre-Training},
  author={Gu, Yuxian and Wen, Jiaxin and Sun, Hao and Song, Yi and Ke, Pei and Zheng, Chujie and Zhang, Zheng and Yao, Jianzhu and Zhu, Xiaoyan and Tang, Jie and Huang, Minlie},
  journal={arXiv preprint arXiv:2203.09313},
  year={2022}
}
```