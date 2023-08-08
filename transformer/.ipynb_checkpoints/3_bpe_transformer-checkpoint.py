import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

from accelerate import Accelerator

import torch
torch.cuda.is_available()
# device = torch.device("cpu")

import wandb
wandb.init()
run_name = wandb.run.name
print(run_name)

base_path = f"""/n/data1/hms/dbmi/beaulieu-jones/lab/epilepsy-transformer"""
model_version = "longformer-model"
input_file = 'input'
vocab_sizek = 52
file_lim = 4000
max_token_length=4096


input_folder = f"{base_path}/input_note/"
# split_var = ['train[:1%]', 'train[-80:]']

#4096 first 1 mil
split_var = [f'train[{training_round-1}0%:{training_round}0%]', 'train[-80:]']
# split_var = ['train[:10%]', 'train[-80:]']
# split_var = ['train[:1000]', 'train[-80:]']
output_comb_dir = f"{base_path}/models/agg-scratch-longformer-{input_file}-{max_token_length}"


from tokenizers import Tokenizer
trained_tok_file = f"{base_path}/tokenizers/bpe_{vocab_sizek}_{file_lim}_{max_token_length}.json"
tokenizer = Tokenizer.from_file(trained_tok_file)

# from tokenizers.processors import TemplateProcessing
# tokenizer.post_processor = TemplateProcessing(
#     single="[CLS] $A [SEP]",
#     pair="[CLS] $A [SEP] $B:1 [SEP]:1",
#     special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
# )
# tokenizer.enable_truncation(max_length=max_token_length)
# tokenizer.enable_padding(length=max_token_length)

from tokenizers.implementations import BaseTokenizer
from transformers import PreTrainedTokenizerFast, AutoTokenizer
fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, model_max_length=max_token_length)
fast_tokenizer.add_special_tokens({'pad_token': '[PAD]', 'mask_token': '[MASK]'})
print(fast_tokenizer)

# tokenizer.get_vocab_size(with_added_tokens=True)
from datasets import load_dataset
# train_dataset = load_dataset("text", 
#                              data_files=f"{input_folder}{input_file}.txt", 
#                              cache_dir="/n/data1/hms/dbmi/beaulieu-jones/lab/transformer_training_data/hugging_face_cache/",
#                              # streaming=True
#                           )

def tokenize_function(examples):
    # Remove empty lines
    examples["text"] = [
        line for line in examples["text"] if len(line) > 0 and not line.isspace()
    ]
    return fast_tokenizer(
        examples["text"],
        padding=False,
        truncation=True,
        max_length=max_token_length-2,
        return_special_tokens_mask=True,
    )

# def tokenize_function(examples):
#     return fast_tokenizer(examples["text"], padding="max_length", truncation=True,
#                           max_length=max_token_length,
#                           pad_to_multiple_of=max_token_length)

train_dataset, test_dataset = load_dataset("text", 
                                           data_files=f"{input_folder}{input_file}.txt", 
                                           split=split_var,
                                           cache_dir=f"{base_path}/hugging_face_cache/")

# train_dataset.filter(lambda example: len(example['text'].split(" ")) >= 10 and len(example['text'].split(" ")) <= 512)
# tokenized_train_dataset = train_dataset.map(tokenize_function, 
#                                  batched=True, 
#                                  num_proc=19)
tokenized_train_dataset = train_dataset.map(tokenize_function, 
                                            num_proc=20,
                                            batched=True)
print('tokenized dataset length', len(tokenized_train_dataset))
# # test_dataset.filter(lambda example: len(example['text'].split(" ")) >= 10)
# tokenized_test_dataset = test_dataset.map(tokenize_function, 
#                                  batched=True, 
#                                  num_proc=2)

# from datasets import Dataset, load_from_disk
# # tokenized_train_dataset=load_from_disk(f"{output_comb_dir}/{input_file}_{split_var_per}.ds")
# tokenized_train_dataset=load_from_disk(f"{output_comb_dir}/{input_file}_test.ds")

from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=fast_tokenizer, mlm=True, mlm_probability=0.15
)

from transformers import Trainer, TrainingArguments
from transformers import LongformerConfig, LongformerModel, LongformerForMaskedLM
# Initializing a Longformer configuration
configuration = LongformerConfig()
configuration.vocab_size = tokenizer.get_vocab_size() #vocab_sizek * 1000
configuration.max_position_embeddings = max_token_length+2
# Initializing a model from the configuration
model = LongformerForMaskedLM(configuration)
# Accessing the model configuration
configuration = model.config
print(configuration)
print('tokenized dataset length', len(tokenized_train_dataset))

training_args = TrainingArguments(
    output_dir=output_comb_dir,
    overwrite_output_dir=False,
    num_train_epochs=1,
    auto_find_batch_size=True,
    save_steps=5_00,
    save_total_limit=1000,
    prediction_loss_only=True,
    report_to="wandb",
    resume_from_checkpoint=True,
    do_eval=False,
    logging_strategy="steps",
    logging_steps=20,
    dataloader_num_workers=8,
    log_level="warning",
    fp16=True,
    lr_scheduler_type='constant_with_warmup',
    gradient_accumulation_steps=4,
    # no_cuda=True,
    # per_device_train_batch_size=1,
    # evaluation_strategy="steps",
    # eval_steps=1_000,
    # fp16_full_eval=True,
    # per_device_eval_batch_size=2,
    # eval_accumulation_steps=1
)


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_train_dataset,
    # eval_dataset=tokenized_test_dataset
)

trainer.train()
trainer.save_model(f"{output_comb_dir}")
