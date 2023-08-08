import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

from sklearn.model_selection import StratifiedKFold
from accelerate import Accelerator

import pandas as pd
import numpy as np
import datasets
from datasets import load_metric, Dataset, DatasetDict

import evaluate
from transformers import DataCollatorWithPadding
import torch
from torch import nn
from transformers import Trainer
from transformers import EarlyStoppingCallback, IntervalStrategy

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

max_token_length=4096

def compute_metrics(eval_pred):
    # predictions = eval_pred.predictions
    # labels = eval_pred.labels
    predictions, labels = eval_pred
    
    roc_auc_metric = load_metric("roc_auc") 
    auc = roc_auc_metric.compute(references=labels, prediction_scores=predictions)['roc_auc']
    
    metric1 = load_metric("precision")
    metric2 = load_metric("recall")
    
    
    # logits, labels = eval_pred
    # predictions = np.argmax(logits, axis=-1)
    binary_pred = np.where(predictions > 0.5, 1, 0)
    
    acc_metric = evaluate.load('accuracy') 
    acc = acc_metric.compute(predictions=binary_pred, references=labels)['accuracy']
    
    
    f1_metric = load_metric("f1") 
    f1 = f1_metric.compute(predictions=binary_pred, references=labels)['f1']
    
    precision = metric1.compute(predictions=binary_pred, references=labels)["precision"]
    recall = metric2.compute(predictions=binary_pred, references=labels)["recall"]

    return {"accuracy": acc, "f1":f1, "auc":auc, "precision": precision, "recall": recall, 
            "pred_positive": sum(binary_pred)[0], "sum_positive": sum(predictions)[0], 
            "actual_positive": sum(labels)}
        
def run_training(in_label, frozen, split_num):
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
        'all_relevant': all_relevant
    }

    key = 'all_relevant'
    base_path = f"""/n/data1/hms/dbmi/beaulieu-jones/lab/epilepsy-transformer/classifier-input"""
    fname = f"{base_path}/{key}_classifier_no_filter_input.csv"

    df = pd.read_csv(fname)
    input_df = df[~df['text'].isna()]

    label_df = pd.read_csv('./data/labels.csv')
    label_df.fillna(0, inplace=True)
    label_df.describe()
    label_cols = label_df.columns

    vocab_sizek = 52
    training_round=2
    # checkpoint = "checkpoint-81000"
    max_token_length=4096

    base_path = f"""/n/data1/hms/dbmi/beaulieu-jones/lab/epilepsy-transformer/models/cat-scratch-longformer"""
    output_comb_dir = f"{base_path}/clean-model"

    # from tokenizers import Tokenizer
    # trained_tok_file = f"{base_path}/tokenizers/bpe_no_filter_{vocab_sizek}_all_relevant.json"
    # tokenizer = Tokenizer.from_file(trained_tok_file)
    # tokenizer.enable_truncation(max_length=max_token_length)
    # print(tokenizer)

       
    from tokenizers.implementations import BaseTokenizer
    from transformers import PreTrainedTokenizerFast, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, model_max_length=max_token_length)
    fast_tokenizer.add_special_tokens({'pad_token': '[PAD]', 'mask_token': '[MASK]'})
    print(fast_tokenizer)

    # model_wcheckpoint = f"{output_comb_dir}/{training_round}/{checkpoint}"
    model_wcheckpoint = f"{output_comb_dir}/{training_round}/"

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained("yikuan8/Clinical-Longformer", 
                                                               num_labels=1)

    label=in_label
    label_df['composite'] = 0.0
    label_df.loc[label_df['structured_drug_class'] > 1, 'composite'] = 1
    label_df.loc[label_df['note_drug_classes'] > 2, 'composite'] = 1
    label_df.loc[label_df['status_epilepticus'] > 0, 'composite'] = 1
    label_df.loc[label_df['inpatient_seizure'] > 0, 'composite'] = 1
    label_df.loc[label_df['seizure_icd_days'] > 4, 'composite'] = 1
    label_df.loc[label_df['seizure_visit_days'] > 4, 'composite'] = 1
    label_df.loc[label_df['neuro_note_count'] > 4, 'composite'] = 1
    label_df.loc[label_df['proc_dates'] > 0, 'composite'] = 1


    # aggregate
    label_df['label'] = 0
    if label == 'composite':
        label_df['label'] = label_df['composite']
    elif label in ['status_epilepticus', 'total_inpatient', 'inpatient_seizure', 'proc_dates']:
        label_df.loc[label_df[label==0] , 'label'] = 0
        label_df.loc[label_df[label>0] , 'label'] = 1
    elif label in ['seizure_icd_days', 'seizure_visit_days', 'structured_drug_class', 'note_drug_classes',
                 'neuro_note_count']:
        label_df.loc[label_df[label<=1] , 'label'] = 0
        label_df.loc[label_df[label>1] , 'label'] = 1
    else:
        raise Exception('stop')
    
    input_df = input_df.merge(label_df, on='patient_num', how='left')
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    split_series = input_df.groupby(['patient_num'])['label'].max()

    print(type(split_series))
    pat_level_df = pd.DataFrame(split_series)
    pat_level_df.reset_index(inplace=True)
    pat_level_df.columns = ['patient_num', 'label']
    
    splits = folds.split(np.zeros(pat_level_df['patient_num'].nunique()), pat_level_df['label'])

    for i in range(split_num+1):
        train_idx, val_idx = next(splits)
    print(len(train_idx), len(val_idx))

    sub_val_idx = val_idx[:500]

    dataset = Dataset.from_pandas(input_df[['text', 'patient_num', 'label']], preserve_index=False)

    print(len(dataset))
    dataset = dataset.filter(lambda example: len(example["text"]) > 20)
    print(len(dataset))

    # train_idxs, val_idxs = enumerate(splits, 0)
    fold_dataset = DatasetDict({
        "train":dataset.filter(lambda example: example['patient_num'] in pat_level_df.iloc[train_idx]['patient_num'].tolist(),
                            num_proc=5),
        "sub_val":dataset.filter(lambda example: example['patient_num'] in pat_level_df.iloc[sub_val_idx]['patient_num'].tolist(),
                                num_proc=5),
        "val": dataset.filter(lambda example: example['patient_num'] in pat_level_df.iloc[val_idx]['patient_num'].tolist(),
                            num_proc=5),
    })

    print('train', len(fold_dataset['train']['label']), sum(fold_dataset['train']['label']))
    print('sub_val', len(fold_dataset['sub_val']['label']), sum(fold_dataset['sub_val']['label']))
    print('val', len(fold_dataset['val']['label']), sum(fold_dataset['val']['label']))

    def tokenize_function(examples):
        # Remove empty lines
        examples["text"] = [
            line for line in examples["text"] if len(line) > 0 and not line.isspace()
        ]
        # return tokenizer.encode(''.join(examples["text"]))
        return fast_tokenizer(
            examples["text"],
            padding=False,
            truncation=True,
            max_length=max_token_length-2,
            return_special_tokens_mask=True,
        )
    
    tokenized_fold_dataset = fold_dataset.map(tokenize_function, num_proc=8, batched=True)

    
    positive_label = tokenized_fold_dataset['train'].filter(lambda example: example['label']==1, num_proc=8) 
    negative_label = tokenized_fold_dataset['train'].filter(lambda example: example['label']==0, num_proc=8)
    print(len(positive_label), len(negative_label))

    seeds = [42, 1234, 4567]
    balanced_data = None
    for s in seeds:
            if balanced_data:
                balanced_data = datasets.concatenate_datasets([balanced_data, 
                                                            datasets.interleave_datasets([
                                                                positive_label.shuffle(seed=s), 
                                                                negative_label.shuffle(seed=s)]
                                                            )]
                                                            )
            else:
                balanced_data = datasets.interleave_datasets([positive_label, negative_label])
    
    print(len(balanced_data), sum(balanced_data['label']))

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model_name = f"orig-{label}-{key}-{frozen}-{split_num}"
    # batch_size = 4
    base_path = f"""/n/data1/hms/dbmi/beaulieu-jones/lab/epilepsy-transformer/models/cat-scratch-longformer"""

    args = TrainingArguments(
        f"{base_path}/finetuned/{model_name}",
        evaluation_strategy = IntervalStrategy.STEPS,
        eval_steps = 100-(50*frozen),
        # max_eval_samples=100,
        # save_strategy = "epoch",
        # auto_find
        per_device_train_batch_size=8+(24*frozen),
        auto_find_batch_size=True,
        learning_rate=2e-5,
        weight_decay=0.01,
        dataloader_num_workers=12,
        # per_device_train_batch_size=4,
        per_device_eval_batch_size=8+(24*frozen),
        gradient_accumulation_steps=1,
        eval_accumulation_steps=1,
        num_train_epochs=5,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_strategy=IntervalStrategy.STEPS,
        logging_steps=10,
        push_to_hub=False,
        fp16=True,
        log_level='warning',
        report_to="wandb",
    )

    for name, param in model.named_parameters():
        if frozen == 1:
            if name.startswith("classifier"):
                param.requires_grad = True
            elif name.startswith("longformer.encoder.layer.11"):
                param.requires_grad = False
            else:    
                param.requires_grad = False
        else: 
            param.requires_grad = True
    
    trainer = Trainer(
        model,
        args,
        train_dataset=balanced_data,
        eval_dataset=tokenized_fold_dataset["sub_val"],
        tokenizer=fast_tokenizer,
        compute_metrics=compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=10)]
    )

    trainer.train()
    trainer.evaluate()
    trainer.evaluate(tokenized_fold_dataset['val'])
    trainer.save_model(f"./data/orig_model_{in_label}_{frozen}_{split_num}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='AL text classifciation')
    
    parser.add_argument('--label', type=str, default='composite',
                        help='label to use')
    parser.add_argument('--frozen', type=int, default=0,
                        help='freeze taransformer?')
    parser.add_argument('--cv_split', type=int, default=0,
                        help='which cv fold')

    args = parser.parse_args()
    print(f"""Input Args:
    label: {args.label}
    frozen: {args.frozen}
    cv_split: {args.cv_split}
    """)

    run_training(args.label, args.frozen, args.cv_split)