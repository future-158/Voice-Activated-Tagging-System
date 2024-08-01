import random
import string
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import torchaudio
from datasets import Audio, Dataset, DatasetDict, load_dataset
from pydub import AudioSegment
from transformers import pipeline
from TTS.api import TTS


class TTSPipe:
    def __init__(self):
        pipe = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)    
        self.pipe = pipe
        
    def text_to_audio(
            self,
            text: str,
            speaker_wav: str,
            file_path: Optional[str] = None,
            language: Optional[str] = 'en'
    ) -> AudioSegment:
        """ return pydub.AudioSegment given text"""
        pipe = self.pipe

        if file_path is None:
            file_path = 'temp-chunk-tts.wav'

        pipe.tts_to_file(
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




ds = load_dataset("openslr/librispeech_asr", "clean")


def generate_random_sequence(length=4):
    characters = string.ascii_lowercase  # You can also include string.ascii_uppercase, string.digits, etc.
    return ''.join(random.choices(characters, k=length))


def audio_segment_to_array(audio_segment):
    samples = np.array(audio_segment.get_array_of_samples())
    if audio_segment.channels > 1:
        samples = samples.reshape((-1, audio_segment.channels))
    # Convert int16 to float32
    samples = samples.astype(np.float32) / 32768.0
    return samples, audio_segment.frame_rate



split2examples = {}
for split in ['train']*500 + ['test']*50 :
    match split:
        case 'train':
            example = ds['train.100'].shuffle().take(1)[0]
        case 'test':
            example = ds['test'].shuffle().take(1)[0]
    
    audio_data = example['audio']['array']
    sample_rate = example['audio']['sampling_rate']
    audio_data_int16 = np.int16(audio_data * 32767)

    # Convert to 16-bit PCM format
    # Write the NumPy array to a WAV file
    output_file = 'output.wav'

    from scipy.io.wavfile import write
    write(output_file, sample_rate, audio_data_int16)

    length = random.randint(3, 10)  
    random_sequence = generate_random_sequence(length)
    sentence = ' '.join([phonetic_alphabet[c] for c in random_sequence])
    segment = tts_pipe.text_to_audio(sentence, speaker_wav=output_file)


    waveform, sample_rate = torchaudio.load(segment.export(format='wav'))
    assert waveform.ndim == 2 # channel, time
    waveform_mono = waveform.mean(dim=0)

    example = dict(
        audio = dict(
        path = None,
        array = waveform_mono.numpy(),
        sampling_rate = sample_rate
    ),
    sentence = sentence
    )
    split2examples.setdefault(split, []).append(example)



dataset = DatasetDict()
dataset['train'] = Dataset.from_list(split2examples['train'])
dataset['test'] = Dataset.from_list(split2examples['test'])


if "":
    # for push to hub. you should replace with your own username
    from datasets import Audio, Dataset, DatasetDict, load_dataset
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset.push_to_hub("famousdetectiveadrianmonk/nato-phoentic-alphabet-voice")



