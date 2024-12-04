# Whisper Fine-tune speech recognition models and speed up reasoning

## Introduction

OpenAI open-sourced project Whisper, which claims to have human-level speech recognition in English, and it also supports automatic speech recognition in 98 other languages. Whisper provides automatic speech recognition and translation tasks. They can turn speech into text in various languages and translate that text into English. The main purpose of this project is to fine-tune the Whisper model using Lora. It supports training on non-timestamped data, with timestamped data, and without speech data. Currently open source for several models, specific can be [openai](https://huggingface.co/openai) to view, the following is a list of commonly used several models. In addition, the project also supports CTranslate2 accelerated reasoning and GGML accelerated reasoning. As a hint, accelerated reasoning supports direct use of Whisper original model transformation, and does not necessarily need to be fine-tuned. Supports Windows desktop applications, Android applications, and server deployments.

### please :star: 

## Supporting models

- openai/whisper-tiny
- openai/whisper-base
- openai/whisper-small
- openai/whisper-medium
- openai/whisper-large
- openai/whisper-large-v2
- openai/whisper-large-v3

**Environment：**

- Miniconda 3
- Python 3.10.2
- Pytorch 2.3.1
- transformers 4.45.2
- Ubuntu 22.04

## Catalogue

- [Install](#安装环境)
- [Prepare data](#准备数据)
- [Fine-tuning](#微调模型)
    - [Single-GPU](#单卡训练)
    - [Multi-GPU](#多卡训练)
- [Merge model](#合并模型)
- [Evaluation](#评估模型)
- [Ctranslate2 inference](#使用Ctranslate2格式模型预测)


<a name='安装环境'></a>

## 安装环境

- Install the required libraries.

```shell
pip install -r requirements.txt
```

<a name='准备数据'></a>

## Prepare the data

The training dataset is a list of jsonlines

**Note:**

1. If timestamp training is not used, the `sentences` field can be excluded from the data.
2. If data is only available for one language, the language field can be excluded from the data.
3. If training empty speech data, the `sentences` field should be `[]`, the `sentence` field should be `""`, and the language field can be absent.
4. Data may exclude punctuation marks, but the fine-tuned model may lose the ability to add punctuation marks.

The format should look like this, but in one line.

```jsonl
{
  "audio": {
    "path": "dataset/0.wav"
  },
  "sentence": "近几年，不但我用书给女儿压岁，也劝说亲朋不要给女儿压岁钱，而改送压岁书。",
  "language": "Chinese",
  "sentences": [
    {
      "start": 0,
      "end": 1.4,
      "text": "近几年，"
    },
    {
      "start": 1.42,
      "end": 8.4,
      "text": "不但我用书给女儿压岁，也劝说亲朋不要给女儿压岁钱，而改送压岁书。"
    }
  ],
  "duration": 7.37
}
```

<a name='微调模型'></a>

## Fine-tune

Once we have our data ready, we are ready to fine-tune our model. Training is the most important two parameters, respectively, `--base_model` specified fine-tuning the Whisper of model, the parameter values need to be in [HuggingFace](https://huggingface.co/openai), the don't need to download in advance, It can be downloaded automatically when starting training, or in advance, if `--base_model` is specified as the path and `--local_files_only`is set to True. The second `--output_path` is the Lora checkpoint path saved during training as we use Lora to fine-tune the model. If you want to save enough, it's best to set `--use_8bit` to False, which makes training much faster. See this program for more parameters.

<a name='单卡训练'></a>

### Single-GPU

The single card training command is as follows. Windows can do this without the `CUDA_VISIBLE_DEVICES` parameter.

```shell
CUDA_VISIBLE_DEVICES=0 python finetune.py --base_model=openai/whisper-tiny --output_dir=output/
```

or Use `run.sh` to use my preset parameters.

<a name='多卡训练'></a>

### Multi-GPU (might not be helpful uwu)

torchrun and accelerate are two different methods for multi-card training, which developers can use according to their preferences.

1. To start multi-card training with torchrun, use `--nproc_per_node` to specify the number of graphics cards to use.

```shell
torchrun --nproc_per_node=2 finetune.py --base_model=openai/whisper-tiny --output_dir=output/
```

log:

```shell
{'loss': 0.9098, 'learning_rate': 0.000999046843662503, 'epoch': 0.01}                                                     
{'loss': 0.5898, 'learning_rate': 0.0009970611012927184, 'epoch': 0.01}                                                    
{'loss': 0.5583, 'learning_rate': 0.0009950753589229333, 'epoch': 0.02}                                                  
{'loss': 0.5469, 'learning_rate': 0.0009930896165531485, 'epoch': 0.02}                                          
{'loss': 0.5959, 'learning_rate': 0.0009911038741833634, 'epoch': 0.03}
```

<a name='合并模型'></a>

## Merge model

You can use `lora_convert_ct2.sh` to convert output model to `ct2`

For SFT model, you can do `convert-ct2.sh`

<a name='评估模型'></a>

## Evaluation

The following procedure is performed to evaluate the model, the most important two parameters are respectively. The first `--model_path` specifies the path of the merged model, but also supports direct use of the original whisper model, such as directly specifying `openai/Whisper-large-v2`, and the second `--metric` specifies the evaluation method. For example, there are word error rate `cer` and word error rate `wer`. Note: Models without fine-tuning may have punctuation in their output, affecting accuracy. See this program for more parameters.

```shell
python evaluation.py --model_path=models/whisper-tiny-finetune --test_data=dataset/test.jsonl
```

<a name='预测'></a>

## Inference

After convert your model to ct2, you can easily reference `live_demo/gradio_demo.py` to build an gradio interface.

<a name='使用Ctranslate2格式模型预测'></a>

## Ctranslate2 inference

As we all know, directly using the Whisper model reasoning is relatively slow, so here provides a way to accelerate, mainly using CTranslate2 for acceleration, first to transform the model, transform the combined model into CTranslate2 model. In the following command, the `--model` parameter is the path of the merged model, but it is also possible to use the original whisper model directly, such as `openai/whisper-large-v2`. The `--output_dir` parameter specifies the path of the transformed CTranslate2 model, and the `--quantization` parameter quantizes the model size. If you don't want to quantize the model, you can drop this parameter.

```shell
ct2-transformers-converter --model models/whisper-tiny-finetune --output_dir models/whisper-tiny-finetune-ct2 --copy_files tokenizer.json preprocessor_config.json --quantization float16
```

Execute the following program to accelerate speech recognition, where the `--audio_path` argument specifies the audio path to predict. `--model_path` specifies the transformed CTranslate2 model. See this program for more parameters.

```shell
python live_demo/gradio_demo.py --model_path=model/finetuned_whisper-ct2
```

## Reference

1. https://github.com/yeyupiaoling/Whisper-Finetune
2. https://huggingface.co/blog/fine-tune-whisper
3. https://github.com/huggingface/peft
4. https://github.com/guillaumekln/faster-whisper
