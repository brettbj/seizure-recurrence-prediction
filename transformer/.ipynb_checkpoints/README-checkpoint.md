# Additional Pre-training for Transformers

- Compare training from scratch to using pre-trained model (both tokenizer and transformer) 
- compare all notes vs. neurology specific notes 

### Steps 

0. 0_PreprocessData - select the notes to be included in the pre-training step. 
1. 1_PrepareNotesForTokenizer - prepare the notes to be used as input training a tokenizer (hugging face)
2. Tokenizer 
- 2_Tokenizer - training a tokenizer from scratch
- Comparing tokenized data to sanity check performance
3. Training Various Transformers 