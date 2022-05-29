import torch
from transformers import BertTokenizer, BartForConditionalGeneration
from transformers import Seq2SeqTrainer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments
from preprocess import Data
from metric import compute
import jieba
import numpy as np
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'


def predict(sentence):
    inputs = tokenizer([sentence], max_length=1024, return_tensors='pt')
    title_ids = model.generate(inputs['input_ids'], num_beams=70, max_length=150, min_length=10, early_stopping=True)
    title = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in title_ids]
    return ' '.join(title)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(jieba.cut(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(jieba.cut(label.strip())) for label in decoded_labels]

    result = compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")
    model = BartForConditionalGeneration.from_pretrained("fnlp/bart-base-chinese")
    # text1text_generator = Text2TextGenerationPipeline(model, tokenizer)

    batch_size = 8
    args = Seq2SeqTrainingArguments(
        'model',
        evaluation_strategy='steps',
        learning_rate=3e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.1,
        save_steps=20,
        save_total_limit=10,
        num_train_epochs=5,
        predict_with_generate=True,
        eval_steps=20,
        logging_dir='logs',
        logging_first_step=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model, padding=True)
    data = Data(tokenizer)
    tokenized_train_dataset, tokenized_test_dataset = data.preProcess()

    trainer = Seq2SeqTrainer(model, args,
                             train_dataset=tokenized_train_dataset,
                             eval_dataset=tokenized_test_dataset,
                             data_collator=data_collator,
                             tokenizer=tokenizer,
                             compute_metrics=compute_metrics)

    trainer.train()
