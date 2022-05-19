# EVA: 大规模中文开放域对话系统

[English version](https://github.com/thu-coai/EVA/blob/main/README_en.md)

## :star2:更新

- 2022.5: 开源 EVA2.0-base 与 EVA2.0-large 模型
- 2022.3: 开源 EVA2.0-xLarge 模型，发布 EVA2.0 的[论文](https://arxiv.org/abs/2203.09313)。
- 2022.1: 开源 fine-tune 代码。
- 2021.8: 开源 EVA1.0 模型及交互代码，发布 EVA1.0 的[论文](https://arxiv.org/abs/2108.01547)。

## 1 项目简介

EVA 是目前最大的开源中文预训练对话模型，拥有28亿参数，主要擅长开放域闲聊，目前有 1.0 和 2.0 两个版本。其中，1.0版本在 [WudaoCorpus-Dialog](https://resource.wudaoai.cn/home) 上训练而成，2.0 版本在从 WudaoCorpus-Dialog 中清洗出的更高质量的对话数据上训练而成，模型性能也明显好于 EVA1.0。EVA1.0 [论文链接](https://arxiv.org/abs/2108.01547)，EVA2.0 [论文链接](https://arxiv.org/abs/2203.09313)。

本仓库中提供了模型交互式评测，模型静态评测，模型微调的代码。

## 2 模型下载

EVA1.0 和 EVA2.0-xLarge 模型可以从[智源下载专区](https://wudaoai.cn/model/detail/EVA)下载，EVA1.0 下载后的目录应该具有如下结构：

```[bash]
eva/
├── 222500
│   └── mp_rank_00_model_states.pt
├── latest_checkpointed_iteration.txt
```

EVA2.0 下载后的目录应该具有如下结构：

```[bash]
eva2/
├── 1
│   └── mp_rank_00_model_states.pt
├── latest_checkpointed_iteration.txt
```

EVA2.0-base 和 EVA2.0-large 模型可以从[此处](https://drive.google.com/drive/folders/1LoEl-j_BGn2gqGMwkXiWCNGjEYBWKeAP?usp=sharing)下载。

## 3 运行

所有代码都包含在 `src/` 目录下.

### 3.1 环境配置

代码运行需要 CUDA10.2 toolkit。交互式评测大约需要占用 7000MB 显存，静态评测和模型微调占用的显存取决于 batch size 和最大输入长度。经过测试，当前微调的超参数配置可以在 4*32G V100 上跑起来。我们提供了两种配置环境的方式。

#### 方式1: 使用 requirements.txt

安装基础依赖

```bash
pip install -r requirements.txt
```

安装 apex

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

安装 deepspeed

我们使用了 `v0.3.9` 版本的 deepspeed，可以从[此仓库](https://github.com/microsoft/DeepSpeed/releases/tag/v0.3.9)中下载安装，或者运行如下命令：

```[bash]
pip install deepspeed==0.3.9
```

由于此版本的 deepspeed 有一些 **bug**，您可能需要对安装后的 python 包做一些修改。关于 bug 的具体信息您可以参考[此问题](https://github.com/TsinghuaAI/CPM-2-Finetune/issues/11) 。简单来说，您需要修改 `${PATH_TO_PYTHON_SITE_PACKAGE}/deepspeed/runtime/zero/stage1.py` 与 `${PATH_TO_PYTHON_SITE_PACKAGE}/deepspeed/runtime/engine.py` 中的几行代码。 我们在仓库中提供了修改后两个文件：`src/ds_fix/stage1.py` 与 `src/ds_fix/engine.py`。您只需要将 `${PATH_TO_PYTHON_SITE_PACKAGE}/deepspeed/runtime/zero/stage1.py` 替换为 `src/ds_fix/stage1.py`，`${PATH_TO_PYTHON_SITE_PACKAGE}/deepspeed/runtime/engine.py` 替换为 `src/ds_fix/engine.py` 即可。

#### 方式2: 使用 Docker

```[bash]
docker pull gyxthu17/eva:1.5
```

因为上述环境已经在 docker 中预装，您不需要再设置任何环境变量了。为了运行代码，您可能需要将此仓库挂在到 docker 中的目录，例如，`/mnt` 目录。为此，您可以运行如下代码：

```[bash]
docker run -ti -v ${PWD}:/mnt gyxthu17/eva:1.5 /bin/bash
```

### 3.2 准备数据

将训练、验证、测试数据放在一个目录下，该目录中需要有 `train.txt`， `valid.txt` 和 `test.txt` 三个文件，文件的每一行为一个对话样本（展开后），轮次之间用`\t`分隔，最后一轮为模型需要生成的对话，前面的轮次为对话上下文，具体格式可以参考我们给出的预处理好的 [KdConv 数据](https://drive.google.com/file/d/1AO06NvhFA5axZci8hC9IZ6nJEYjkbRKg/view?usp=sharing)。KdConv 原始数据可从[此仓库](https://github.com/thu-coai/KdConv)下载。

### 3.3 运行代码

所有运行脚本都在 `src/scripts` 中。

+ 交互式评测脚本：`eva_inference_interactive_beam.sh` 与 `eva_inference_interactive_no_beam.sh`
+ 静态评测脚本：`eva_inference_static.sh`
+ 微调脚本：`eva_finetune.sh`

在运行以上脚本之前，需要先将 `WORKING_DIR` 改为此 EVA 目录的路径, 将 `CKPT_PATH` 改为存储预训练 checkpoint 的路径。静态评测和微调还需要将`DATA_PATH`改为3.2中的数据目录，该目录下需要有 `train.txt`， `valid.txt` 和 `test.txt` 三个文件，训练/评测结果存储位置`SAVE_PATH`也可以按照需求修改。其它参数含义可以参考中 `eva_finetune.sh` 的注释。

**注意**：EVA2.0 与 EVA1.0 在模型结构上有一些差别，在更换模型时请注意同时更换模型配置文件。项目中默认提供 EVA2.0-xLarge 的模型配置文件：`eva2.0_model_config.json`，EVA1.0 的配置文件为 `eva1.0_model_config.json`。更改执行脚本中的 `CONFIG_PATH` 即可。

上述修改修改完成后运行：

```[bash]
cd src/
bash scripts/eva_inference_interactive_beam.sh # 交互式评测，使用 beam search 解码
bash scripts/eva_inference_interactive_no_beam.sh # 交互式评测，不使用 beam search 解码
bash scripts/eva_inference_static.sh # 静态评测
bash scripts/eva_finetune.sh # 微调模型
```

**注意**：运行上述命令后, 您需要确定预训练模型加载成功。如果它们加载成功，stdout 中会输出 `successfully loaded /path-to-checkpoint/eva/mp_rank_01_model_states.pt`. 否则，会输出 `WARNING: could not find the metadata file /***/latest_checkpointed_iteration.txt will not load any checkpoints and will start from random`。需要注意的是，当成功加载模型后，程序还会输出 `The following zero checkpoints paths are missing: ['/path-to-checkpoint/eva/200000/zero_pp_rank_0_mp_rank_00_optim_states.pt',...` 一大串 log，说明没有加载优化器的参数。因为本仓库代码只进行评测和微调，是否加载优化器参数没有影响，所以您可以忽略这个 log。

如果上述脚本正常运行，对于交互式评测，您会看到一个交互提示符，可以在后面输入文字和 EVA 对话，输入 `clear` 可以重新开始对话，输入 `seed` 可以设置随机种子。对于静态评测和模型微调，代码会读取数据并启动模型训练和推理，最后的结果存储在 `SAVE_PATH` 中。

### 3.4 修改模型并行度

如果发现单卡显存不够，可以使用 `src/change_mp.py` 更改模型并行度，下面命令中 `TARGET_MP` 表示目标模型并行度。下载下来的模型并行度为1，即所有模型参数在用一张 GPU 上，可以通过增加并行度将一个模型的参数分摊到多个卡上，从而减少单张卡的显存占用（当然，这可能意味着需要用更多的卡来训练）。注意对下载的模型修改之后还要将训练/推理脚本中的 `MP_SIZE` 修改为对应的并行度。

```[bash]
cd src/
python3 change_mp.py ${INPUT_MODEL_DIR} ${OUTPUT_MODEL_DIR} ${TARGET_MP}
```

## 4 参考结果

我们使用处理好的 KdConv 数据集进行评测，按照仓库中给出的超参数微调、测试集上静态评测得到如下结果

|                        |      |        |        |       |
| ----                   | ---- | ----   | ----   | ----  |
|                        | loss | BLEU-4 | Dist-4 | f1    |
| eva1.0 Finetune 前     | 3.49 | 0.64   | 85.94  | 13.24 |  
| eva1.0 Finetune 后     | 2.37 | 3.94   | 72.47  | 19.80 |
| eva2.0 Finetune 前     | 3.14 | 1.02   | 82.67  | 14.36 |  
| eva2.0 Finetune 后     | 2.09 | 4.69   | 73.03  | 22.91 |

使用 beam search 对 EVA2.0 模型进行交互式评测，我们获得了如下样例。

```[]
Usr >>> 你好
Sys >>> 你好，请问你是？
Usr >>> 我是小明
Sys >>> 好的，小明，我是新来的，请多关照
Usr >>> 你了解人工智能吗？
Sys >>> 不了解，我只知道我是机器人
Usr >>> 那你很聪明
Sys >>> 谢谢夸奖，我会继续努力的
Usr >>> 你是学什么专业的呀
Sys >>> 计算机科学与技术，你呢？
Usr >>> 我也是学计算机的
Sys >>> 哦，那你有什么想知道的，可以问我
Usr >>> 编程语言你学过哪些？
Sys >>> java,c,c+都学过
Usr >>> 你学的真多
Sys >>> 呵呵，还好吧，我比较喜欢写代码
Usr >>> 哈哈，我也喜欢写代码，以后可以多多交流
Sys >>> 嗯，好的
Usr >>> 拜拜喽
Sys >>> 拜～
```

**注意**：由于不同硬件的随机函数可能不同，即使使用和我们相同的随机种子，可能仍然无法复现样例结果。但是整体性能应该不会有太大差距。

## 5 免责声明

本预训练对话模型仅限科研用途。模型训练数据集中的对话收集自不同的来源，虽然我们设计了一套严格的数据清洗流程，但是我们并不保证所有不当内容均已被过滤。该数据中所包含的所有内容和意见与本项目作者无关。 本项目所提供的模型和代码仅为完整对话系统的一个组成部分，我们所提供的解码脚本仅限科研用途，使用本项目中的模型和脚本所生成的一切对话内容与本项目作者无关。

## 6 TODO

+ ~~finetune 代码整理与开源~~
+ ~~EVA2.0 模型下载链接~~
+ ~~EVA2.0 技术报告~~
+ ~~开源小规模模型~~
+ huggingface 版本的模型/对应代码
+ 预训练数据处理代码开源

## 7 引用

```[]
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
