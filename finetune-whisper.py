if "":
    !pip install --upgrade --quiet pip
    !pip install --upgrade --quiet datasets[audio] transformers accelerate evaluate jiwer tensorboard gradio


from datasets import load_dataset, DatasetDict, Dataset



if "":
    common_voice = DatasetDict()
    common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="train+validation", use_auth_token=True, trust_remote_code=True)
    common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="test", use_auth_token=True, trust_remote_code=True)
    common_voice = common_voice.select_columns(["audio", "sentence"])  



common_voice = DatasetDict()
common_voice["train"] = Dataset.load_from_disk("data/dataset_500/train")    
common_voice["test"] = Dataset.load_from_disk("data/dataset_500/test")


common_voice = common_voice.select_columns(["audio", "sentence"])  

from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small", task="transcribe")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", task="transcribe")
processor = WhisperProcessor.from_pretrained("openai/whisper-small", task="transcribe")

print(common_voice["train"][0])

from datasets import Audio
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

"""We can apply the data preparation function to all of our training examples using dataset's `.map` method. The argument `num_proc` specifies how many CPU cores to use. Setting `num_proc` > 1 will enable multiprocessing. If the `.map` method hangs with multiprocessing, set `num_proc=1` and process the dataset sequentially."""

common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"])

common_voice['train'][0]['labels']
common_voice['train'][0]['input_features']

from transformers import WhisperForConditionalGeneration
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")


# model.generation_config.language = "en"
# model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None

"""### Define a Data Collator

The data collator for a sequence-to-sequence speech model is unique in the sense that it
treats the `input_features` and `labels` independently: the  `input_features` must be
handled by the feature extractor and the `labels` by the tokenizer.

The `input_features` are already padded to 30s and converted to a log-Mel spectrogram
of fixed dimension by action of the feature extractor, so all we have to do is convert the `input_features`
to batched PyTorch tensors. We do this using the feature extractor's `.pad` method with `return_tensors=pt`.

The `labels` on the other hand are un-padded. We first pad the sequences
to the maximum length in the batch using the tokenizer's `.pad` method. The padding tokens
are then replaced by `-100` so that these tokens are **not** taken into account when
computing the loss. We then cut the BOS token from the start of the label sequence as we
append it later during training.

We can leverage the `WhisperProcessor` we defined earlier to perform both the
feature extractor and the tokenizer operations:
"""

import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

"""Let's initialise the data collator we've just defined:"""

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

"""### Evaluation Metrics

We'll use the word error rate (WER) metric, the 'de-facto' metric for assessing
ASR systems. For more information, refer to the WER [docs](https://huggingface.co/metrics/wer). We'll load the WER metric from 🤗 Evaluate:
"""

import evaluate
metric = evaluate.load("wer")

"""We then simply have to define a function that takes our model
predictions and returns the WER metric. This function, called
`compute_metrics`, first replaces `-100` with the `pad_token_id`
in the `label_ids` (undoing the step we applied in the
data collator to ignore padded tokens correctly in the loss).
It then decodes the predicted and label ids to strings. Finally,
it computes the WER between the predictions and reference labels:
"""

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

"""### Define the Training Configuration

In the final step, we define all the parameters related to training. For more detail on the training arguments, refer to the Seq2SeqTrainingArguments [docs](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments).
"""

from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-hi",  # change to a repo name of your choice
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

"""**Note**: if one does not want to upload the model checkpoints to the Hub,
set `push_to_hub=False`.

We can forward the training arguments to the 🤗 Trainer along with our model,
dataset, data collator and `compute_metrics` function:
"""

from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

"""We'll save the processor object once before starting training. Since the processor is not trainable, it won't change over the course of training:"""

trainer.train()

processor.save_pretrained(training_args.output_dir)

tokenizer.save_pretrained(training_args.output_dir)
feature_extractor.save_pretrained(training_args.output_dir)

trainer.save_model(training_args.output_dir)



from transformers import pipeline
import gradio as gr




from transformers import WhisperForConditionalGeneration
model = WhisperForConditionalGeneration.from_pretrained(training_args.output_dir)   


feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small", task="transcribe")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", task="transcribe")
processor = WhisperProcessor.from_pretrained("openai/whisper-small", task="transcribe")


from transformers import pipeline
whisper_asr = pipeline(
    "automatic-speech-recognition", model=model, device=0, tokenizer=tokenizer ,
    feature_extractor=feature_extractor
)


whisper_norm = whisper_asr.tokenizer._normalize







"""We then set our batch size. We also restrict the number of samples for evaluation to 128 for the purpose of this blog. If you want to run on the full dataset to get the official results, comment out or remove this line from the proceeding code cell!"""


# only for debugging, restricts the number of rows to numeric value in brackets

def normalise(batch):
    batch["sentence"] = whisper_norm(batch['sentence'])
    return batch


import numpy as np
def data(dataset):
    for i, item in enumerate(dataset):
        dic =  {**item["audio"], "reference": item["sentence"]}
        dic['array'] = np.array(dic['array'])
        yield dic   



test_ds = common_voice["test"] = Dataset.load_from_disk("data/dataset_500/test")
dataset = test_ds.map(normalise)




predictions = []
references = []

# run streamed inference
for out in whisper_asr(data(dataset)):
    
    prediction = whisper_norm(out["text"])
    
    predictions.append(prediction)
    references.append(out["reference"][0])

    print('-'*50)
    print(prediction)
    print(out["reference"][0])


import evaluate
wer_metric = evaluate.load("wer")


"""We can now pass on our list of references and predictions to the WER evaluate function to compute the WER:"""
wer = wer_metric.compute(references=references, predictions=predictions)
wer = round(100 * wer, 2)

print("WER:", wer) # 

# 1시간 학습후 wer 2.76 기존 5에서 반절로 떨어짐




def transcribe(audio):
    text = pipe(audio)["text"]
    return text

iface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(source="microphone", type="filepath"),
    outputs="text",
    title="Whisper Small Hindi",
    description="Realtime demo for Hindi speech recognition using a fine-tuned Whisper small model.",
)

iface.launch()

"""## Closing Remarks

In this blog, we covered a step-by-step guide on fine-tuning Whisper for multilingual ASR
using 🤗 Datasets, Transformers and the Hugging Face Hub. For more details on the Whisper model, the Common Voice dataset and the theory behind fine-tuning, refere to the accompanying [blog post](https://huggingface.co/blog/fine-tune-whisper). If you're interested in fine-tuning other
Transformers models, both for English and multilingual ASR, be sure to check out the
examples scripts at [examples/pytorch/speech-recognition](https://github.com/huggingface/transformers/tree/main/examples/pytorch/speech-recognition).
"""