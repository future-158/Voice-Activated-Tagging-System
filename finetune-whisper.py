from datasets import load_dataset, DatasetDict, Dataset
from datasets import Audio

common_voice = load_dataset("famousdetectiveadrianmonk/nato-phoentic-alphabet-voice")
common_voice = common_voice.select_columns(["audio", "sentence"])  
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained("openai/whisper-small", task="transcribe")
feature_extractor = processor.feature_extractor
tokenizer = processor.tokenizer

print(common_voice["train"][0])


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"])


common_voice['train'][0]['input_features']
common_voice['train'][0]['labels']

from transformers import WhisperForConditionalGeneration
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

# i dont know what this option does, and at this point i am too afraid to ask
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
    output_dir="./whisper-small-en",  # change to a repo name of your choice
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=2000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=500,
    eval_steps=500,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)


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
trainer.save_model(training_args.output_dir)

if "":
    # push to hub. use your repo name
    model.push_to_hub("famousdetectiveadrianmonk/whisper-small-nato-phoentic-alphabet")




from transformers import pipeline
import gradio as gr
from transformers import WhisperForConditionalGeneration
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

if "":
    model = WhisperForConditionalGeneration.from_pretrained("famousdetectiveadrianmonk/whisper-small-nato-phoentic-alphabet")   
else:
    model = WhisperForConditionalGeneration.from_pretrained(training_args.output_dir)   


processor = WhisperProcessor.from_pretrained("openai/whisper-small", task="transcribe")
model = AutoModelForSpeechSeq2Seq.from_pretrained("famousdetectiveadrianmonk/whisper-small-nato-phoentic-alphabet")


from transformers import pipeline
whisper_asr = pipeline(
    "automatic-speech-recognition", model=model, device=0, tokenizer=processor.tokenizer ,
    feature_extractor=processor.feature_extractor
)

# i dont know what this option does, and at this point i am too afraid to ask
whisper_norm = whisper_asr.tokenizer._normalize



def normalise(batch):
    batch["sentence"] = whisper_norm(batch['sentence'])
    return batch


import numpy as np
def data(dataset):
    for i, item in enumerate(dataset):
        dic =  {**item["audio"], "reference": item["sentence"]}
        dic['array'] = np.array(dic['array'])
        yield dic   



test_ds = load_dataset("famousdetectiveadrianmonk/nato-phoentic-alphabet-voice", split='test')
test_ds = test_ds.select_columns(["audio", "sentence"])  
test_ds = test_ds.cast_column("audio", Audio(sampling_rate=16000))
test_ds  = test_ds.map(normalise)


predictions = []
references = []
for out in whisper_asr(data(test_ds)):
    prediction = whisper_norm(out["text"])
    predictions.append(prediction)

    references.append(out["reference"][0])

    import random
    if random.random() < 0.1:
        print('-'*50)
        print(out["reference"][0])
        print(prediction)


import evaluate
wer_metric = evaluate.load("wer")


"""We can now pass on our list of references and predictions to the WER evaluate function to compute the WER:"""
wer = wer_metric.compute(references=references, predictions=predictions)
wer = round(100 * wer, 2)

print("WER:", wer) # 




