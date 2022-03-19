# EVA: An Open-Domain Chinese Dialogue System with Large-Scale Generative Pre-Training

[中文版](https://github.com/thu-coai/EVA)

## :star2:Update

- March 2022: Release the EVA2.0 pre-trained model and [our paper](https://arxiv.org/abs/2203.09313) 
- Jan 2022: Release the fine-tuning code.
- Aug 2021: Release the EVA1.0 pre-trained model, the interacting code, and [our paper](https://arxiv.org/abs/2108.01547).

## 1 Introduction

EVA is the largest open-source Chinese dialogue model with up to 2.8B parameters. The 1.0 version model is pre-trained on [WudaoCorpus-Dialog](https://resource.wudaoai.cn/home), and the 2.0 version is pre-trained on a carefully cleaned version of WudaoCorpus-Dialog which yields better performance than the 1.0 version. [Paper link](https://arxiv.org/abs/2108.01547) of EVA1.0. [Paper link](https://arxiv.org/abs/2203.09313) of EVA2.0.

We provide the interactive inference, static inference, and finetuning code of EVA in this repo.

## 2 Model Download
EVA1.0 and EVA2.0 model can be downloaded from [BAAI repository](https://wudaoai.cn/model/detail/EVA), the downloaded directory of EVA1.0 should look like this:

```[bash]
eva/
├── 222500
│   └── mp_rank_00_model_states.pt
├── latest_checkpointed_iteration.txt
```

The downloaded directory of EVA2.0 should look like this:

```[bash]
eva2/
├── 1
│   └── mp_rank_00_model_states.pt
├── latest_checkpointed_iteration.txt
```

## 3 Run the Code

The source code is provided in `src/`.

### 3.1 Environment

The code requires the CUDA10.2 toolkit. The interactive inference occupies only about 7000MB of GPU memory. The memory consumption of static inference and model finetuning depends on the batch size and the max input length. We find that 4*32 V100 is enough for the code to run under the default hyperparameters in the scripts. We provide 2 options to set up the environment.

#### Option 1: Local setup

##### Install basic dependencies

```bash
pip install -r requirements.txt
```

##### Install apex

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
##### Install DeepSpeed

The version we used is `v0.3.9`, It can be installed from its [repo](https://github.com/microsoft/DeepSpeed/releases/tag/v0.3.9) or 
```bash
pip install deepspeed==0.3.9
```
Since there exist some **bugs** in DeepSpeed, you need to make some little modifications to this package. You can refer to this [issue](https://github.com/TsinghuaAI/CPM-2-Finetune/issues/11) for more information. Specifically, you need to modify two lines of code in `${PATH_TO_PYTHON_SITE_PACKAGE}/deepspeed/runtime/zero/stage1.py` and `${PATH_TO_PYTHON_SITE_PACKAGE}/deepspeed/runtime/engine.py`. We provide the modified `src/ds_fix/stage1.py` and `src/ds_fix/engine.py` in our repo. You can simply replace `${PATH_TO_PYTHON_SITE_PACKAGE}/deepspeed/runtime/zero/stage1.py` with `stage1.py` and `${PATH_TO_PYTHON_SITE_PACKAGE}/deepspeed/runtime/engine.py` with `engine.py` that we provided. 

#### Option 2: Docker

```[bash]
docker pull gyxthu17/eva:1.5
```

Since the environment is ready in the docker, you don't need to set any environment variables. You may need to mount this directory to a directory in the docker. For example, to mount to /mnt, run the following code to run the docker image:

```[bash]
docker run -ti -v ${PWD}:/mnt gyxthu17/eva:1.5 /bin/bash
```

### 3.2 Prepare Data

To prepare the data, you need to put training, validation and, test datasets under a directory, which contains `train.txt`, `valid.txt`, and `test.txt`. Each line in the files is a dialogue sample (after expanding). Different utterances are separated with `\t` and the last utterance is the response that the model needs to generate. The specific format can be referred to the preprocessed [KdConv data](https://drive.google.com/file/d/1AO06NvhFA5axZci8hC9IZ6nJEYjkbRKg/view?usp=sharing) the original data can be obtained from this [repo](https://github.com/thu-coai/KdConv).


### 3.3 Run

All the scripts to run is in `src/scripts`.

+ Interactive inference：`eva_inference_interactive_beam.sh` and `eva_inference_interactive_no_beam.sh`
+ Static inference：`eva_inference_static.sh`
+ Finetuning：`eva_finetune.sh`

Before running the code, please change `WORKING_DIR` in the script to the path of this EVA directory, change `CKPT_PATH` to the path where the pre-trained weights are stored. For static inference and finetuning, you need to change `DATA_PATH` to the data path in section 3.2, which contains `train.txt`, `valid.txt`, and `test.txt`. You can specify where to store the results by modifying `SAVE_PATH`. The explanations of other parameters can be found in `eva_finetune.sh`.

**NOTE:** The model architecture of EVA2.0 and EVA1.0 is slightly different. Therefore, please change the model configuration file by modifying `CONFIG_PATH` if you switch the base model. 

Then run the following command:

```[bash]
cd src/
bash scripts/eva_inference_interactive_beam.sh # interactive inference with beam search
bash scripts/eva_inference_interactive_no_beam.sh # interactive inference without beam search
bash scripts/eva_inference_static.sh # static inference
bash scripts/eva_finetune.sh # finetune the model
```

**NOTE**: After running the command, please first make sure the pre-trained weights are loaded. If they are loaded, the log printed to the stdout should contain messages like `successfully loaded /path-to-checkpoint/eva/mp_rank_01_model_states.pt`. Otherwise, `WARNING: could not find the metadata file /***/latest_checkpointed_iteration.txt will not load any checkpoints and will start from random` will display. Note that when you successfully load the model, you will see messages like `The following zero checkpoints paths are missing: ['/path-to-checkpoint/eva/200000/zero_pp_rank_0_mp_rank_00_optim_states.pt',...` which mean optimizer states are not loaded. This **DOES NOT** affect the use of model inference and you can just ignore it.

If things go well, for interactive inference, you will eventually enter an interactive interface. You can chat with EVA by typing after `>>>`. When you input `clear`, the dialogue history will be cleared and the conversation will start over. When you input `seed`, you can change the random seed. 

For static inference and finetuning, the code will read the data, start training or inference, and the results can be found in `SAVE_PATH`.

### 3.4 Change the Model Parallism Size

If you find that the memory of a single GPU is not large enough, you can try using `src/change_mp.py` to change the model parallelism size. The `TARGET_MP` in the following command represents the target model parallelism size. The model parallelism size of the downloaded model is 1, which means all model parameters are stored on a single GPU. You can increase the model parallelism size to distribute the parameters to multi GPUs, which can reduce the memory requirement of a single GPU (Of course, you may need more GPUs to train the model). Note that after changing the parallelism size of the downloaded model, you also need to change the `MP_SIZE` in the training/inference scripts to the corresponding model parallelism size.

```[bash]
cd src/
python3 change_mp.py ${INPUT_MODEL_DIR} ${OUTPUT_MODEL_DIR} ${TARGET_MP}
```

## 4 Results

We use the processed KdConv dataset to finetune and evaluate the model. Under the default hyperparameters in the scripts, we get the following results on the test set:

|                        |      |        |        |       |
| ----                   | ---- | ----   | ----   | ----  |
|                        | loss | BLEU-4 | Dist-4 | f1    |
| eva1.0 Before finetune     | 3.49 | 0.64   | 85.94  | 13.24 |  
| eva1.0 After finetune    | 2.37 | 3.94   | 72.47  | 19.80 |
| eva2.0 Before finetune     | 3.14 | 1.02   | 82.67  | 14.36 |  
| eva2.0 After finetune     | 2.09 | 4.69   | 73.03  | 22.91 |

We also use beam search to run the inferactive inference of EVA2.0:

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

**NOET**: Since different hardware may have different implementations of random functions, even if you use the same random seed as ours, you may not be able to reproduce this case. But the overall performance will not differ much.

## 5 Disclaimer

The pre-trained models aim to facilitate the research for conversation generation. The model provided in this repository is trained on a large dataset collected from various sources. Although a rigorous cleaning and filtering process has been carried out to the data and the model output, there is no guarantee that all the inappropriate contents have been completely banned. All the contents generated by the model do not represent the authors' opinions. The decoding script provided in this repository is only for research purposes. We are not responsible for any content generated using our model.


## 6 TODO

+ ~~Open source code for finetuning~~
+ ~~EVA2.0 model download link.~~
+ ~~EVA2.0 technical report.~~
+ Provide model/codes of in hugginface style.
+ Models with small sizes.
+ The code to process pre-training data.

## 7 Citation

```[]
@article{coai2021eva,
  title={EVA: An Open-Domain Chinese Dialogue System with Large-Scale Generative Pre-Training},
  author={Zhou, Hao and Ke, Pei and Zhang, Zheng and Gu, Yuxian and Zheng, Yinhe and Zheng, Chujie and Wang, Yida and Wu, Chen Henry and Sun, Hao and Yang, Xiaocong and Wen, Bosi and Zhu, Xiaoyan and Huang, Minlie and Tang, Jie},
  journal={arXiv preprint arXiv:2108.01547},
  year={2021}
}
@article{coai2022eva2,
  title={EVA2.0: Investigating Open-Domain Chinese Dialogue Systems with Large-Scale Pre-Training},
  author={Yuxian Gu, Jiaxin Wen, Hao Sun, Yi Song, Pei Ke, Chujie Zheng, Zheng Zhang, Jianzhu Yao, Xiaoyan Zhu, Jie Tang, Minlie Huang},
  journal={arXiv preprint arXiv:2108.01547},
  year={2022}
}
```
