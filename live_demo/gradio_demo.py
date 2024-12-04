import sys

from faster_whisper import WhisperModel
from transformers import pipeline
import gradio as gr

from utils.utils import print_arguments, add_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("model_path",  type=str, default="models/whisper-tiny-finetune-ct2", help="converted ct2 model")
add_arg("base_model",  type=str, default="openai/whisper-small")
add_arg("language",    type=str, default="zh", help="set transcribe language. if None, it will auto detect language")
add_arg("device", type=str, default="cpu")
add_arg("without_timestamps",  type=bool, default=True,    help="use timestamps or no")
add_arg("beam", type=int, default=5)
add_arg("temperature", type=float, default=0)
add_arg("repetition_penalty", type=float, default=1.1)
args = parser.parse_args()
print_arguments(args)

# load models
model = WhisperModel(args.model_path, device=args.device, compute_type="auto")

if args.language != None:
    tokenizer = WhisperTokenizer.from_pretrained(args.base_model, language=args.language, task="transcribe")
else:
    tokenizer = WhisperTokenizer.from_pretrained(args.base_model)

original_model = pipeline(
    "automatic-speech-recognition",
    tokenizer=tokenizer,
    model=args.base_model,
    device=args.device
)

# get model base name as Gradio title
if args.model_path[-1] == '/':
    model_name = args.model_path[:-1]
model_name = model_name.split('/')[-1]
print(model_name)


# transcribe loop
def transcribe(audio_path, audio):
    # Transcribe Finetuned model
    if audio_path != '':
        audio = audio_path
    utt = audio.split('/')[-1]
    print(f'Processing: {utt}')
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
    text = ''
    for seg in segments:
        text += seg.text

    orig_text = original_model(audio)["text"]

    return (text, orig_text)


with gr.Blocks() as demo:
    gr.HTML(f'<div style="display:inline; text-align:center"><h1>Hokkien {model_name}</h1></div>')
    with gr.Row():
        with gr.Column(scale=4):
            audio_path = gr.Textbox(label="audio path (prior)")
            audio = gr.Audio(sources=["microphone", "upload"], type="filepath", show_download_button=True)
            submit_btn = gr.Button(value="Submit")
        with gr.Column(scale=2):
            output = gr.Textbox(label="output", interactive=False)
            orig_output = gr.Textbox(label="original whisper output", interactive=False)
    with gr.Row():
        submit_btn.click(
            transcribe,
            inputs=[audio_path, audio],
            outputs=[output, orig_output],
            api_name=False
        )

demo.queue().launch(
    share=True,
    server_name='0.0.0.0',
    server_port=19324
)
