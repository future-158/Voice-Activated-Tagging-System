from datasets import load_dataset
ds = load_dataset("openslr/librispeech_asr", "clean")


import gc
import os
import random
import random
import string
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import numpy as np
import torch
import torchaudio
import tqdm
from datasets import Dataset, DatasetDict
from more_itertools import chunked
from pyannote.audio import Pipeline
from pydub import AudioSegment
from transformers import (AutoModelForCausalLM, AutoModelForSpeechSeq2Seq,
                          AutoProcessor, AutoTokenizer, pipeline)
from TTS.api import TTS


class TTSPipe:
    def __init__(self):
        pipe = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)    
        # self.__call__ = _wrap_with_pipe_call(self.__call__, pipe.tts_to_file)
        self.pipe = pipe
        
    # def __call__(self, *args: torch.Any, **kwds: torch.Any) -> torch.Any:
    #     return self.pipe(*args, **kwds)

    # def __getattr__(self, name):
    #     return getattr(self.pipe, name)
    
    def text_to_audio(
            self,
            text: str,
            speaker_wav: str,
            file_path: Optional[str] = None,
            language: Optional[str] = 'en'
    ) -> AudioSegment:
        """ return pydub.AudioSegment given text"""
        if file_path is None:
            file_path = 'temp-chunk-tts.wav'
            
        self.pipe.tts_to_file(
            text=text,
            file_path=file_path,
            speaker_wav=speaker_wav,
            language=language
            )

        segment = AudioSegment.from_file(file_path)
        Path(file_path).unlink(missing_ok=True)
        return segment
    

tts_pipe = TTSPipe()

phonetic_alphabet = {
    "a": "alpha",
    "b": "bravo",
    "c": "charlie",
    "d": "delta",
    "e": "echo",
    "f": "foxtrot",
    "g": "golf",
    "h": "hotel",
    "i": "india",
    "j": "juliet",
    "k": "kilo",
    "l": "lima",
    "m": "mike",
    "n": "november",
    "o": "oscar",
    "p": "papa",
    "q": "quebec",
    "r": "romeo",
    "s": "sierra",
    "t": "tango",
    "u": "uniform",
    "v": "victor",
    "w": "whiskey",
    "x": "x-ray",
    "y": "yankee",
    "z": "zulu"
}


style_files = [ p for p in Path('voice-style').glob('*.wav')]

from datasets import load_dataset


def generate_random_sequence(length=4):
    characters = string.ascii_lowercase  # You can also include string.ascii_uppercase, string.digits, etc.
    return ''.join(random.choices(characters, k=length))


import uuid
def export_audio(audio_segment) -> str:
    dest = Path('data/segments') / '{}.wav'.format(uuid.uuid4().hex)
    dest.parent.mkdir(parents=True, exist_ok=True)
    audio_segment.export(dest, format='wav')
    return dest.absolute().as_posix()





common_voice = DatasetDict()
# common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="train+validation", use_auth_token=True, trust_remote_code=True)
common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="test[:5%]", use_auth_token=True, trust_remote_code=True)
common_voice = common_voice.select_columns(["audio", "sentence"])  


common_voice = Dataset.load_from_disk("data/dataset_50").train_test_split(test_size=0.1)   

from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small", task="transcribe")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", task="transcribe")
processor = WhisperProcessor.from_pretrained("openai/whisper-small", task="transcribe")



print(common_voice["train"][0])

"""Since
our input audio is sampled at 48kHz, we need to _downsample_ it to
16kHz prior to passing it to the Whisper feature extractor, 16kHz being the sampling rate expected by the Whisper model.

We'll set the audio inputs to the correct sampling rate using dataset's
[`cast_column`](https://huggingface.co/docs/datasets/package_reference/main_classes.html?highlight=cast_column#datasets.DatasetDict.cast_column)
method. This operation does not change the audio in-place,
but rather signals to `datasets` to resample audio samples _on the fly_ the
first time that they are loaded:
"""

from datasets import Audio
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))






# Example usage


ds['train.100'][0]['text']
ds['train.100'][0]['audio']
ds['test']









import torchaudio
torchaudio.load('voice-style/voice-style-bae-60s.wav')[0]



import torch
import tempfile



import numpy as np
from scipy.io.wavfile import write


import numpy as np
from pydub import AudioSegment
import os
import numpy as np
from pydub import AudioSegment
from datasets import Dataset


def audio_segment_to_array(audio_segment):
    # Get the raw audio data as an array of samples
    samples = np.array(audio_segment.get_array_of_samples())
    
    # If the audio has multiple channels, the samples array will be interleaved.
    # Reshape the array to separate the channels
    if audio_segment.channels > 1:
        samples = samples.reshape((-1, audio_segment.channels))
    
    return samples, audio_segment.frame_rate


audio_array, sampling_rate = audio_segment_to_array(audio_segment)


def load_audio_file(file_path):
    audio_segment = AudioSegment.from_wav(file_path)
    samples, sample_rate = audio_segment_to_array(audio_segment)
    return {
        "path": file_path,
        "array": samples,
        "sampling_rate": sample_rate
    }

def audio_segment_to_array(audio_segment):
    samples = np.array(audio_segment.get_array_of_samples())
    if audio_segment.channels > 1:
        samples = samples.reshape((-1, audio_segment.channels))
    # Convert int16 to float32
    samples = samples.astype(np.float32) / 32768.0
    return samples, audio_segment.frame_rate

def create_dataset(audio_paths, sentences):
    data = []
    for audio_path, sentence in zip(audio_paths, sentences):
        audio_data = load_audio_file(audio_path)
        data.append({
            "audio": audio_data,
            "sentence": sentence
        })
    return Dataset.from_list(data)





example = ds['train.100'].shuffle().take(1)[0]


audio_data = example['audio']['array']
sample_rate = example['audio']['sampling_rate']

audio_data_int16 = np.int16(audio_data * 32767)

# Convert to 16-bit PCM format

# Write the NumPy array to a WAV file
output_file = 'output.wav'
write(output_file, sample_rate, audio_data_int16)

audio_segment = AudioSegment.from_wav("output.wav")


split2examples = {}

for split in ['train']*500 + ['test']*50 :
    # style_file = random.choice(style_files)

    match split:
        case 'train':
            example = ds['train.100'].shuffle().take(1)[0]
        case 'test':
            example = ds['test'].shuffle().take(1)[0]
    # raise
    audio_data = example['audio']['array']
    sample_rate = example['audio']['sampling_rate']
    audio_data_int16 = np.int16(audio_data * 32767)

    # Convert to 16-bit PCM format
    # Write the NumPy array to a WAV file
    output_file = 'output.wav'
    write(output_file, sample_rate, audio_data_int16)

    audio_segment = AudioSegment.from_wav("output.wav")

    # raise
    import random
    length = random.randint(3, 10)  

    random_sequence = generate_random_sequence(length)
    sentence = ' '.join([phonetic_alphabet[c] for c in random_sequence])
    segment = tts_pipe.text_to_audio(sentence, speaker_wav=output_file)

    filepath = export_audio(segment)
    # torchaudio.load(segment.export(format='wav'))[0]
    waveform, sample_rate = torchaudio.load(filepath)
    example = dict(
        audio = dict(
        path = filepath,
        array = waveform.squeeze().numpy(),
        sampling_rate = sample_rate
    ),
    sentence = sentence
    )
    split2examples.setdefault(split, []).append(example)
    
dataset = DatasetDict()
dataset['train'] = Dataset.from_list(split2examples['train'])
dataset['test'] = Dataset.from_list(split2examples['test'])





idx = random.randint(0 , len(dataset['train'])-1)
example = dataset['train'][idx]
audio_segment = AudioSegment.from_file(example['audio']['path'])
print(example['sentence'])
audio_segment

dataset['train'].save_to_disk("data/dataset_500/train")    
dataset['test'].save_to_disk("data/dataset_500/test")



# zeroshot test




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


dataset = dataset.map(normalise)

dataset[0]['audio']
dataset[0]['norm_text']


from datasets import Audio
dataset = dataset.cast_column("audio", Audio(sampling_rate=32000))



def data(dataset):
    for i, item in enumerate(dataset):
        yield {**item["audio"], "reference": item["norm_text"]}

"""We then set our batch size. We also restrict the number of samples for evaluation to 128 for the purpose of this blog. If you want to run on the full dataset to get the official results, comment out or remove this line from the proceeding code cell!"""


# only for debugging, restricts the number of rows to numeric value in brackets


predictions = []
references = []

# run streamed inference
for out in whisper_asr(data(dataset)):
    prediction = whisper_norm(out["text"])
    predictions.append(prediction)
    references.append(out["reference"][0])

    print(prediction)

"""We can now pass on our list of references and predictions to the WER evaluate function to compute the WER:"""
wer = wer_metric.compute(references=references, predictions=predictions)
wer = round(100 * wer, 2)


print("WER:", wer)