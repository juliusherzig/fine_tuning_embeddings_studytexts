"""
Purpose: Fine-tuning the ModernBERT-model with SetFit for text classification between study texts of replicable and non-replicable studies. 
To prevent exceeding the text length limit, the study texts had been separeted into 4 equal-sized parts. For each part i, a single model is fine-tuned. 
"""

# Package-Imports
import argparse
import pandas as pd
from datasets import Dataset, ClassLabel
from setfit import SetFitModel, Trainer, TrainingArguments
from dotenv import load_dotenv
from transformers import EarlyStoppingCallback 


load_dotenv()

def train_part(i):
    """Train a single model for a single text segment (1-4)."""
    print(f"--- START TRAINING FOR PART {i} ---")

    model = SetFitModel.from_pretrained("nomic-ai/modernbert-embed-base", trust_remote_code=True)
    print("Loading model successful!")

    # Load JSON text files and create a Pandas DataFrame
    df = pd.read_json(f"data/studytextPart{i}.jsonl", lines=True)

    df = df.rename(columns={  # rename columns
        "text": "text",
        "replicationSuccessSigDir": "label",
        "numberOriginal": "numberOriginal",
        "setfitSplit": "split"
    })

    print(f"Data loaded:\n{df.head()}")

    ################################ Datasplit ################################
    # Use the variable split that was created in R in advance to ensure the same data split in all 4 text segments.

    train_df = df[df['split'] == "train"]
    test_df = df[df['split'] == "test"]
    val_df = df[df['split'] == "val"] #new: for Early Stopping: validation set separated from training set for stepwise evaluation and Early Stopping

    print(f"Class distribution in the training set:\n{train_df['label'].value_counts()}")
    print(f"Class distribution in the test set:\n{test_df['label'].value_counts()}")
    print(f"Class distribution in the validation set:\n{val_df['label'].value_counts()}")

    # transform datasests into Hugging Face Datasets
    train_dataset = Dataset.from_pandas(train_df) 
    val_dataset = Dataset.from_pandas(val_df)   
    test_dataset = Dataset.from_pandas(test_df)
    
    # Transfrorm lable into ClassLabel
    class_label = ClassLabel(num_classes=2, names=["0", "1"])
    train_dataset = train_dataset.cast_column("label", class_label)
    val_dataset = val_dataset.cast_column("label", class_label)
    test_dataset = test_dataset.cast_column("label", class_label)
    
    del df, train_df, test_df, val_df  # free memory

    ############ Fine-Tuning with SetFit ############
    # Trainingsargumente
    # batch_size is device-specific (16 für GPU, 8 für CPU)
    # use_amp=True für mixed precision at GPU
    batch_size = 4
    steps_interval = 40 // batch_size

    args = TrainingArguments(
        num_epochs = 1,
        batch_size = batch_size, 
        use_amp = True, # On GPU: Mixed precision for faster training
        sampling_strategy = "undersampling",
        eval_strategy = "steps", #new: Early Stopping. Requires this evaluation during training. The idea is to detect when only training loss improves, but prediction on validation set no longer does, indicating overfitting. Then Early Stopping is triggered to stop the training.
        save_strategy = "steps", #new: stepwise saving is necessary to load the previous model when Early Stopping is triggered
        logging_strategy= "steps",
        save_steps = steps_interval, # evaluation during Early Stopping always after 40 training pairs => after 40/batch-size training steps
        eval_steps = steps_interval, 
        logging_steps = steps_interval,
        load_best_model_at_end = True,
        metric_for_best_model="accuracy",
        greater_is_better=True
    )   

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # The validation dataset is the independent, new part of the sample were the classification is evaluated as  early stopping (split from the training set)
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)], #new: Early Stopping prevents overfitting by stopping training when the performance on the validation set stops improving.
        metric="accuracy",  # Evaluate based on accuracy (hit rate)
    )

    # Train the model
    trainer.train()

    ## Save model locally
    output_dir = f"mein_modernbert_studien_modell_{i}"
    model.save_pretrained(output_dir)


def main():
    """Main Function """
    parser = argparse.ArgumentParser() #create empty argument parser  
    parser.add_argument("--part", type=int, choices=[1, 2, 3, 4]) # option to specify the part number (1-4) to be processed. If no part is specified, all parts will be processed.
    args = parser.parse_args()

    if args.part:
        train_part(args.part)
    else:
        for i in range(1, 5):
            train_part(i)

main()