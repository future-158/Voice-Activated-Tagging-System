from datasets import load_dataset, DatasetDict, Dataset
from transformers import pipeline
import gradio as gr
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from transformers import WhisperProcessor


processor = WhisperProcessor.from_pretrained("openai/whisper-small", task="transcribe")



model = AutoModelForSpeechSeq2Seq.from_pretrained("famousdetectiveadrianmonk/whisper-small-nato-phoentic-alphabet")

if "":
    # if you want to use locally saved model
    model = AutoModelForSpeechSeq2Seq.from_pretrained("...")


from transformers import pipeline
whisper_asr = pipeline(
    "automatic-speech-recognition", model=model, device=0, tokenizer=processor.tokenizer ,
    feature_extractor=processor.feature_extractor
)

def transcribe(audio):
    text = whisper_asr(audio)["text"]
    return text

iface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(sources="microphone", type="filepath"),
    outputs="text",
    title="Whisper small nato phonetic alphabet",
    description="This is a demo of the whisper model fine-tuned on the nato phonetic alphabet. Speak into the microphone",
)

iface.launch()

