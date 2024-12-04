import argparse
import functools
import gc
import json
import os

import evaluate
import numpy as np
from tqdm import tqdm
from faster_whisper import WhisperModel, BatchedInferencePipeline

from utils.utils import print_arguments, add_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("test_data",   type=str, default="dataset/test.jsonl",            help="测试集的路径")
add_arg("model_path",  type=str, default="models/whisper-tiny-finetune", help="合并模型的路径，或者是huggingface上模型的名称")
add_arg("language",    type=str, default="zh", help="设置语言，可全称也可简写，如果为None则评估的是多语言")
add_arg("without_timestamps",  type=bool, default=True,    help="评估时是否使用时间戳数据")
add_arg("local_files_only",  type=bool,  default=True, help="是否只在本地加载模型，不尝试下载")
add_arg("beam", type=int, default=5)
add_arg("temperature", type=float, default=0)
add_arg("repetition_penalty", type=float, default=1.1)
add_arg("metric",     type=str, default="cer",        choices=['cer', 'wer'])
args = parser.parse_args()
print_arguments(args)


def main():
    metric = evaluate.load(args.metric)
    model = WhisperModel(args.model_path, device="auto", compute_type="auto")

    if args.model_path[-1] == '/':
        args.model_path = args.model_path[:-1]
    model_name = args.model_path.split('/')[-1]
    if args.test_data[-1] == '/':
        args.test_data = args.test_data[:-1]
    data_name = args.test_data.split('/')[-1][:-6]

    if os.path.isfile(f'dataset/{data_name}_{model_name}.csv'):
        output_result = open(f'dataset/{data_name}_{model_name}.csv', 'a', encoding='utf-8')
    else:
        output_result = open(f'dataset/{data_name}_{model_name}.csv', 'w', encoding='utf-8')
        output_result.write('path,reference,prediction,cer,confidence\n')

    reference = []
    prediction = []
    data = []
    with open(args.test_data) as file:
        for line in tqdm(file):
            data.append(json.loads(line))
    for line in tqdm(data):
        audio = line['audio']['path']
        if not os.path.isfile(audio):
            print(f'{audio} not found, skipping it.')
            continue
        ref = line['sentence']
        segments, info = model.transcribe(
            audio=audio,
            language=args.language,
            beam_size=args.beam,
            best_of=5,
            patience=1,
            repetition_penalty=args.repetition_penalty,
            temperature=args.temperature,
            compression_ratio_threshold=2.0,
            without_timestamps=args.without_timestamps,
        )
        pred = ''
        for seg in segments:
            pred += seg.text
        confidence = seg.avg_logprob
        try:
            cer = 100 * metric.compute(predictions=[pred], references=[ref])
            prediction.append(pred)
            reference.append(ref)
            audio = '\"' + audio + '\"'
            pred = '\"' + pred + '\"'
            ref = '\"' + ref + '\"'
            output_result.write(f'{audio},{ref},{pred},{cer},{confidence}\n')
        except Exception as err:
            print(err)
            continue

    cer = 100 * metric.compute(predictions=prediction, references=reference)
    output_result.write(f'null,null,null,{cer},null\n')

if __name__ == '__main__':
    main()
