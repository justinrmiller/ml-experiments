from datasets import load_dataset

from transformers import ViTForImageClassification, ViTImageProcessor
from transformers import TrainingArguments, Trainer

import evaluate

import torch

import numpy as np

import sys

if len(sys.argv) - 1 != 2:
    print("usage: python model_beans_train.py <batch_size: 16> <epochs: 4>")
    sys.exit(1)

batch_size = int(sys.argv[1])
epochs = int(sys.argv[2])
model_name = "vit-base-beans"

ds = load_dataset('beans')

model_name_or_path = 'google/vit-base-patch16-224-in21k'

image_processor = ViTImageProcessor.from_pretrained(model_name_or_path)


def process_example(example):
    inputs = image_processor(example['image'], return_tensors='pt')
    inputs['labels'] = example['labels']
    return inputs


print(process_example(ds['train'][0]))


def transform(example_batch):
    # Take a list of PIL images and turn them to pixel values
    inputs = image_processor([x for x in example_batch['image']], return_tensors='pt')

    # Don't forget to include the labels!
    inputs['labels'] = example_batch['labels']
    return inputs


prepared_ds = ds.with_transform(transform)


def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }


metric = evaluate.load("accuracy")


def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)


labels = ds['train'].features['labels'].names


model = ViTForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)}
)

use_mps_device = False
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")
else:
    print("Using MPS device")
    use_mps_device = True

training_args = TrainingArguments(
    output_dir=f"./{model_name}",
    per_device_train_batch_size=batch_size,
    evaluation_strategy="steps",
    num_train_epochs=epochs,
    fp16=False,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    learning_rate=2e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    #   report_to='tensorboard',
    load_best_model_at_end=True,
    use_mps_device=use_mps_device,
    optim="adamw_torch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=prepared_ds["train"],
    eval_dataset=prepared_ds["validation"],
    tokenizer=image_processor,
)

print("Entering training phase")

train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()


#
# print("Entering evaluation phase")
#
# metrics = trainer.evaluate(prepared_ds['validation'])
# trainer.log_metrics("eval", metrics)
# trainer.save_metrics("eval", metrics)
