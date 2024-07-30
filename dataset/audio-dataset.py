
# !add-apt-repository -y ppa:jonathonf/ffmpeg-4 && apt update && apt install -y ffmpeg
if "":
    !pip install --quiet datasets transformers evaluate huggingface_hub jiwer

from datasets import load_dataset

dataset = load_dataset("mozilla-foundation/common_voice_17_0", "en", split="test")

usecols = ['audio', 'sentence']
dataset = dataset.select_columns(usecols)



from datasets import Audio
dataset = dataset.cast_column("audio", Audio(sampling_rate=32000))


"""Great! Now we can take a listen of what the audio sounds like and print the text:"""

import IPython.display as ipd
sample = next(iter(dataset))
audio = sample["audio"]

ipd.Audio(data=audio["array"], autoplay=True, rate=audio["sampling_rate"])
print(sample["sentence"])


import os
import torchaudio
import numpy as np
from datasets import Dataset, DatasetDict

def load_audio_file(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    return {
        "path": file_path,
        "array": waveform.squeeze().numpy(),
        "sampling_rate": sample_rate
    }



load_audio_file(
sample['audio']['path']
)

from pydub import AudioSegment
audio_segment = AudioSegment.from_file(sample['audio']['path'])
audio_segment


torchaudio.load(audio_segment.export(format='wav'))[0]




















"""We then define a 'helper function' that gets the correct transcription column from our dataset. We'll use this function to automatically get the right column names when we perform multi-dataset evaluation."""

def get_text(sample):
    if "text" in sample:
        return sample["text"]
    elif "sentence" in sample:
        return sample["sentence"]
    elif "normalized_text" in sample:
        return sample["normalized_text"]
    elif "transcript" in sample:
        return sample["transcript"]
    else:
        raise ValueError(f"Sample: {sample.keys()} has no transcript.")


from transformers import pipeline
whisper_asr = pipeline(
    "automatic-speech-recognition", model="openai/whisper-tiny.en", device=0
)


import evaluate
wer_metric = evaluate.load("wer")


whisper_norm = whisper_asr.tokenizer._normalize

def normalise(batch):
    batch["norm_text"] = whisper_norm(get_text(batch))
    return batch

"""We then use 🤗 Datasets' [`map`](https://huggingface.co/docs/datasets/v2.6.1/en/process#map) method to apply our normalising function across the entire dataset:"""

dataset = dataset.map(normalise)

dataset[0]['audio']
dataset[0]['norm_text']

"""We need to remove any empty reference transcriptions from our dataset, as these will give a divide by 0 error in the WER calculation.

We write a function that indicates which samples to keep, and which to discard. This function, `is_target_text_in_range`, returns a boolean: reference transcriptions that are not empty return True, and those are empty return False:
"""

def is_target_text_in_range(ref):
    if ref.strip() == "ignore time segment in scoring":
        return False
    else:
        return ref.strip() != ""

"""We can apply this filtering function to all of our training examples using 🤗 Datasets' [`filter`](https://huggingface.co/docs/datasets/process#select-and-filter)
method, keeping all references that are not empty (True) and discarding those that are (False):
"""

dataset = dataset.filter(is_target_text_in_range, input_columns=["norm_text"])

"""## Single Dataset Evaluation

Since we're in streaming mode, we won't run inference in place, but rather signal to 🤗 Datasets to perform inference _on the fly_ when the dataset is iterated.

We first define a generator that iterates over the dataset and yields the audio samples and reference text:
"""

def data(dataset):
    for i, item in enumerate(dataset):
        yield {**item["audio"], "reference": item["norm_text"]}

"""We then set our batch size. We also restrict the number of samples for evaluation to 128 for the purpose of this blog. If you want to run on the full dataset to get the official results, comment out or remove this line from the proceeding code cell!"""




import evaluate
wer_metric = evaluate.load("wer")




# only for debugging, restricts the number of rows to numeric value in brackets
dataset = dataset.take(128)

predictions = []
references = []

# run streamed inference
for out in whisper_asr(data(dataset)):
    predictions.append(whisper_norm(out["text"]))
    references.append(out["reference"][0])

"""We can now pass on our list of references and predictions to the WER evaluate function to compute the WER:"""
wer = wer_metric.compute(references=references, predictions=predictions)
wer = round(100 * wer, 2)

print("WER:", wer)





