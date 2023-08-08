# -*- coding: utf-8 -*-
from tokenizers import Tokenizer, processors, normalizers
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace, BertPreTokenizer
from tokenizers.normalizers import BertNormalizer, NFD, StripAccents, Replace, Strip
from pathlib import Path

vocab_sizek = 52
file_lim = 4000
max_token_length=1024

OUTPUT_DIR = '/n/data1/hms/dbmi/beaulieu-jones/lab/epilepsy-transformer' 

paths = [str(x) for x in Path(f"{OUTPUT_DIR}/preproc_note/").glob("**/*.txt")]
print(paths[:file_lim])
files = paths[:file_lim]

from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()
tokenizer.enable_truncation(max_length=max_token_length)
# tokenizer.enable_padding(length=max_token_length)
tokenizer.normalizer = normalizers.Sequence([NFD(), BertNormalizer(), Strip(), Replace("¿", "")])


from tokenizers.processors import TemplateProcessing
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
)

special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
tokenizer.train(files=paths[:file_lim], vocab_size=vocab_sizek*1000, min_frequency=10, show_progress=True, special_tokens=special_tokens)
save_filename = f"{OUTPUT_DIR}/tokenizers/bpe_{vocab_sizek}_{file_lim}_{max_token_length}.json"
print(save_filename)
tokenizer.save(save_filename)

