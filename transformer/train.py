# get the population who have mri + eeg during narrow period
# use to predict cortical dysplasia 
from sqlalchemy import create_engine
import pandas as pd

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

from accelerate import Accelerator

neurology_notes = ['NOTE:125942859', 'NOTE:2820514', 'NOTE:4127998', 'NOTE:4127189', 'NOTE:3817544', 'NOTE:4776343', 'NOTE:67621192', 'NOTE:4127201', 'NOTE:4128000', 'NOTE:4127729', 'NOTE:4127835', 'NOTE:4127923', 'NOTE:4293224', 'NOTE:4127989', 'NOTE:4127595', 'NOTE:4776342', 'NOTE:4293221', 'NOTE:4633365', 'NOTE:4127190', 'NOTE:4127226', 'NOTE:4127203', 'NOTE:4127515', 'NOTE:4127192', 'NOTE:4127524', 'NOTE:4127586', 'NOTE:4293218', 'NOTE:3827999', 'NOTE:4127522', 'NOTE:4127837', 'NOTE:4127194', 'NOTE:599038309', 'NOTE:4127597', 'NOTE:4127999', 'NOTE:4293225', 'NOTE:4127992', 'NOTE:3828001', 'NOTE:4293226', 'NOTE:3755582', 'NOTE:164478027', 'NOTE:125943072', 'NOTE:4776258', 'NOTE:4127227', 'NOTE:599038319', 'NOTE:4127202', 'NOTE:490360275', 'NOTE:4127457', 'NOTE:22652867', 'NOTE:16187985', 'NOTE:400739364', 'NOTE:4776344', 'NOTE:4776072', 'NOTE:4127995', 'NOTE:4128007', 'NOTE:4127204', 'NOTE:4127191', 'NOTE:4127195']
discharge_notes = ['NOTE:3710478', 'NOTE:67621090', 'NOTE:15612283', 'NOTE:15585515', 'NOTE:3710477', 'NOTE:15587049', 'NOTE:15587406', 'NOTE:15612287', 'NOTE:189094536', 'NOTE:15588370', 'NOTE:15588192', 'NOTE:67621087', 'NOTE:15586337', 'NOTE:153048767', 'NOTE:15587712', 'NOTE:189094539', 'NOTE:97195725', 'NOTE:618403281', 'NOTE:122778232']
eeg_notes = ['NOTE:3691367', 'NOTE:3691278', 'NOTE:4120166', 'NOTE:4120168', 'NOTE:3691374', 'NOTE:4120169']
mri_notes = ['NOTE:83328334', 'NOTE:83328424', 'NOTE:3447280', 'NOTE:83328328', 'NOTE:83328617', 'NOTE:83328304', 'NOTE:64217783', 'NOTE:251612637', 'NOTE:251612643', 'NOTE:83328337', 'NOTE:3447282', 'NOTE:83328313', 'NOTE:64217795', 'NOTE:83328427', 'NOTE:83328433', 'NOTE:3447264', 'NOTE:3447296', 'NOTE:64217779', 'NOTE:64217573', 'NOTE:8007688', 'NOTE:3447291', 'NOTE:202592750', 'NOTE:83328370', 'NOTE:83328406', 'NOTE:3447318', 'NOTE:83328325', 'NOTE:64217811', 'NOTE:83328403', 'NOTE:64217538', 'NOTE:83328436', 'NOTE:83328322', 'NOTE:4106734', 'NOTE:83328385', 'NOTE:4106510', 'NOTE:64217571', 'NOTE:64217568', 'NOTE:3447321', 'NOTE:3447275', 'NOTE:3447320', 'NOTE:8007690', 'NOTE:64217799', 'NOTE:83328448', 'NOTE:64217642', 'NOTE:83328292', 'NOTE:251612640', 'NOTE:3447290', 'NOTE:3447301', 'NOTE:3447317', 'NOTE:83328621', 'NOTE:83328412', 'NOTE:83328367', 'NOTE:3447300', 'NOTE:333730849', 'NOTE:3447289', 'NOTE:11921317', 'NOTE:3447293', 'NOTE:83328382', 'NOTE:100343890', 'NOTE:4106495', 'NOTE:83328280', 'NOTE:83328286', 'NOTE:8007698', 'NOTE:3447292', 'NOTE:83328400', 'NOTE:3447306', 'NOTE:8007683', 'NOTE:64217638', 'NOTE:3447266', 'NOTE:3447283', 'NOTE:345927837', 'NOTE:83328379', 'NOTE:83328331', 'NOTE:3447322', 'NOTE:83328373', 'NOTE:3447323', 'NOTE:64217409', 'NOTE:83328363', 'NOTE:64217323', 'NOTE:3447260', 'NOTE:3447313', 'NOTE:64217561', 'NOTE:8007708', 'NOTE:83328298', 'NOTE:8007672', 'NOTE:83328430', 'NOTE:3447308', 'NOTE:3447285', 'NOTE:345927816', 'NOTE:3447314', 'NOTE:4106731', 'NOTE:333730843', 'NOTE:83328301', 'NOTE:64217416', 'NOTE:64217539', 'NOTE:64217666', 'NOTE:64217530', 'NOTE:64217393', 'NOTE:64217384', 'NOTE:8007725', 'NOTE:3447302', 'NOTE:11921316', 'NOTE:3447258']
ct_notes = ['NOTE:3446996', 'NOTE:3447017', 'NOTE:202592738', 'NOTE:3446983', 'NOTE:3446993', 'NOTE:3447020', 'NOTE:3446980', 'NOTE:3447006', 'NOTE:3446997', 'NOTE:3446998', 'NOTE:4873824', 'NOTE:3446982', 'NOTE:230601754', 'NOTE:3446974', 'NOTE:202592729', 'NOTE:3447012', 'NOTE:3446985', 'NOTE:3446999', 'NOTE:64217632', 'NOTE:4873823', 'NOTE:3447011', 'NOTE:3447004', 'NOTE:230601760', 'NOTE:428074121', 'NOTE:244897858', 'NOTE:64217494', 'NOTE:3446990', 'NOTE:463436530', 'NOTE:64217627', 'NOTE:3446991', 'NOTE:64217513', 'NOTE:64217581', 'NOTE:3447015', 'NOTE:8007643', 'NOTE:3447008', 'NOTE:428074112', 'NOTE:452620691', 'NOTE:64217599', 'NOTE:83328241', 'NOTE:3446979', 'NOTE:3447018', 'NOTE:64217357', 'NOTE:3446995']
all_relevant = neurology_notes + discharge_notes + eeg_notes + mri_notes + ct_notes

note_cat_dict = {
    'neurology_notes': neurology_notes,
    'discharge_notes': discharge_notes,
    'eeg_notes': eeg_notes,
    'mri_notes': mri_notes,
    'ct_notes': ct_notes,
    # 'all_relevant': all_relevant
}

base_path = f"""/n/data1/hms/dbmi/beaulieu-jones/lab/epilepsy-transformer/models/cat-scratch-longformer"""

key = 'all_relevant'
# fname = f"{base_path}/training_no_filter_{key}.csv"

output_comb_dir = f"{base_path}/cohort"
print(output_comb_dir)


vocab_sizek = 52

from tokenizers.pre_tokenizers import Whitespace, BertPreTokenizer
from tokenizers.normalizers import BertNormalizer, NFD, StripAccents, Replace, Strip

from tokenizers import Tokenizer
trained_tok_file = f"{base_path}/tokenizers/bpe_no_filter_{vocab_sizek}_{key}.json"
tokenizer = Tokenizer.from_file(trained_tok_file)

max_token_length=4096
from tokenizers.implementations import BaseTokenizer
from transformers import PreTrainedTokenizerFast, AutoTokenizer
fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, model_max_length=max_token_length)
fast_tokenizer.add_special_tokens({'pad_token': '[PAD]', 'mask_token': '[MASK]'})
print(fast_tokenizer)

from datasets import load_dataset

training_round = 1
split_var = [f'train[0%:95%]', 'train[-500:]']
# split_var = [f'train[5%:{training_round}0%]', 'train[-80:]']
# split_var = ['train[:10%]', 'train[-1%:]']

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
 
    
train_dataset, test_dataset = load_dataset("text", 
                                           data_files= f"{base_path}/cohort/notes.txt",
                                           split=split_var,
                                           cache_dir=f"{base_path}/hugging_face_cache/"
                                          )

train_dataset = train_dataset.filter(lambda example: len(example["text"]) > 20)
tokenized_train_dataset = train_dataset.map(tokenize_function, 
                                            num_proc=20,
                                            batched=True)

print('tokenized dataset length', len(tokenized_train_dataset))
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=fast_tokenizer, mlm=True, mlm_probability=0.15
)

from transformers import Trainer, TrainingArguments
from transformers import LongformerConfig, LongformerModel, LongformerForMaskedLM
# Initializing a Longformer configuration
configuration = LongformerConfig()
configuration.vocab_size = tokenizer.get_vocab_size() #vocab_sizek * 1000
configuration.max_position_embeddings = max_token_length
# Initializing a model from the configuration
model = LongformerForMaskedLM(configuration)

# model_string = f"{output_comb_dir}/{training_round}/checkpoint-23500"
# print(model_string)

from transformers import AutoModelForMaskedLM
checkpoint = "checkpoint-102500"
# model = AutoModelForMaskedLM.from_pretrained(f"{output_comb_dir}/{training_round}/{checkpoint}")
# #model = AutoModelForMaskedLM.from_pretrained(f"{output_comb_dir}/")
# print(model.num_parameters())

# Accessing the model configuration
configuration = model.config
print(configuration)
print('tokenized dataset length', len(tokenized_train_dataset))

lr = 0.00005-(0.00001*(training_round-1))
print(lr)
training_args = TrainingArguments(
    output_dir=f"{output_comb_dir}/{training_round}",
    overwrite_output_dir=False,
    num_train_epochs=1,
    optim="adafactor",
    per_device_train_batch_size=2,
    #auto_find_batch_size=True,
    save_steps=5_00,
    save_total_limit=1000,
    prediction_loss_only=True,
    report_to="wandb",
    resume_from_checkpoint=True,
    do_eval=False,
    logging_strategy="steps",
    logging_steps=10,
    # dataloader_num_workers=12,
    log_level="warning",
    fp16=True,
    #gradient_checkpointing=True,
    lr_scheduler_type='constant',
    learning_rate= lr,
    gradient_accumulation_steps=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_train_dataset,
    # eval_dataset=tokenized_test_dataset
)

trainer.train()
# trainer.train(resume_from_checkpoint=f"{output_comb_dir}/{training_round}/{checkpoint}")
# trainer.train(resume_from_checkpoint=True)
trainer.save_model(f"{output_comb_dir}/{training_round}")