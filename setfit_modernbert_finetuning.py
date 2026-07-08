# Package-Imports
import argparse
import os
import pandas as pd
from datasets import Dataset, ClassLabel
from setfit import Trainer, TrainingArguments
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import EarlyStoppingCallback 

from utils.gpu_utils import get_device_config, load_model, get_training_args, logger #written by Ivo Zilkenat to detect GPU/CPU and adapt trainings_args like batch_size depending on that

load_dotenv()
HF_REPO_PREFIX = "juliusherzig/setfit-modernbert-studien-modell"


def train_part(config, i):
    """Train a single model for a single text segment (1-4)."""
    logger.info(f"--- START TRAINING FOR PART {i} ---")

    model = load_model(config)
    logger.info("Loading model successful!")

    # Load JSON text files and create a Pandas DataFrame
    df = pd.read_json(f"data/studytextPart{i}.jsonl", lines=True)

    df = df.rename(columns={  # rename columns
        "text": "text",
        "replicationSuccessSigDir": "label",
        "numberOriginal": "numberOriginal",
        "setfitSplit": "split"
    })

    logger.info(f"Data loaded:\n{df.head()}")

    ################################ Datasplit ################################
    # Use the variable split that was created in R in advance to ensure the same data split in all 4 text segments.

    train_df = df[df['split'] == "train"]
    test_df = df[df['split'] == "test"]
    val_df = df[df['split'] == "val"] #new for Early Stopping: validation set separated from training set for stepwise evaluation and Early Stopping

    logger.info(f"Class distribution in the training set:\n{train_df['label'].value_counts()}")
    logger.info(f"Class distribution in the test set:\n{test_df['label'].value_counts()}")
    logger.info(f"Class distribution in the validation set:\n{val_df['label'].value_counts()}")

    # transform datasests into Hugging Face Datasets
    test_dataset = Dataset.from_pandas(test_df)
    train_dataset = Dataset.from_pandas(train_df) 
    val_dataset = Dataset.from_pandas(val_df)   

    # Transfrorm lable into ClassLabel
    class_label = ClassLabel(num_classes=2, names=["0", "1"])
    train_dataset = train_dataset.cast_column("label", class_label)
    val_dataset = val_dataset.cast_column("label", class_label)
    test_dataset = test_dataset.cast_column("label", class_label)
    
    del df, train_df, test_df

    ############ Fine-Tuning with SetFit ############
    # Trainingsargumente
    # batch_size is device-specific (16 für GPU, 8 für CPU)
    # use_amp=True für mixed precision at GPU
    args = TrainingArguments(
        num_epochs = 1,
        batch_size = config.batch_size, #device-specific (GPU/ CPU)
        use_amp = config.use_amp, #device-specific (GPU/ CPU)
        sampling_strategy = "undersampling",
        eval_strategy = "steps", #new: was previously "no" => no validation set.
        #eval_strategy "steps" => Evaluation during training. Detection of when only training loss improves, but prediction on validation set no longer does, indicating overfitting. Then Early Stopping is triggered to stop the training.
        save_strategy = "steps", #new: necessary to be able to load the previous model when Early Stopping is triggered, because further training does not bring any improvement anymore
        logging_strategy= "steps",
        save_steps =40 // config.batch_size, # evaluation always after 20 training pairs => after 20/batch_size steps
        eval_steps = 40//config.batch_size, 
        logging_steps = 40//config.batch_size,
        load_best_model_at_end = True,
        metric_for_best_model="accuracy",
        greater_is_better=True
    )   

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # Validation dataset! (Split from the training set)
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)], #new: Early Stopping prevents overfitting by stopping training when the performance on the validation set stops improving.
        metric="accuracy",  # Evaluate based on accuracy (hit rate)
    )

    # Train the model
    trainer.train()

    # 6. Evaluate
    #val_results = trainer.evaluate()
    #logger.info(f"Evaluation at the validation set in part {i}: {val_results}")

    #trainer.eval_dataset = test_dataset  # Testset for the final evaluation
    #test_results = trainer.evaluate()
    #logger.info(f"Testset Evaluationsergebnisse Part {i}: {test_results}")

    ## Save model locally
    model.save_pretrained(f"mein_modernbert_studien_modell_{i}")


def main():
    """Main Function """
    parser = argparse.ArgumentParser(description="SetFit ModernBERT Finetuning")
    parser.add_argument("--part", type=int, choices=[1, 2, 3, 4],
                        help="Specific Part (1-4). Without specification, all parts will be trained.")
    args = parser.parse_args()

    # HuggingFace login for model push
    hf_token = os.getenv("HF_TOKEN_WRITE")
    if hf_token:
        login(token=hf_token)
        logger.info("HuggingFace login successful with HF_TOKEN_WRITE")
    else:
        logger.warning("HF_TOKEN_WRITE not set - models will only be saved locally")

    # 1. Set trainning paarameters depending on GPU/CPU 
    config = get_device_config()

    if args.part:
        train_part(config, args.part)
    else:
        for i in range(1, 5):
            train_part(config, i)


if __name__ == "__main__":
    main()