import torch
import time
import numpy as np
import jiwer
import os
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import logging
import wandb
from datetime import datetime

# Import the simple dataset and model
from dataset import load_simple_esposalles_datasets, collate_fn
from model import SimpleEsposallesTrOCR

# Keep the same interface as your original training
def load_datasets(augmentation=False):
    """
    Load datasets with the same interface as your original code
    
    Args:
        augmentation: Whether to apply augmentation to training data
    
    Returns:
        train_dataset, test_dataset
    """
    return load_simple_esposalles_datasets(augmentation=augmentation)

# Environment setup
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Commented out - set via environment or command line
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Choose device based on availability
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
logger.info(f"Using device: {device}")

# Training hyperparameters - UPDATED TO MATCH FIRST CODE
BATCH_SIZE = 6
LR = 5e-6
EPOCHS = 50
EARLY_STOP = 50
NUM_WORKERS = 4
STEP_SIZE = 5
WEIGHT_DECAY = 1e-2

def create_run_folders(run_name=None):
    """Create unique folders for each run based on timestamp or run name."""
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"
    
    # Create base directories if they don't exist
    base_log_folder = 'logs'
    
    if not os.path.exists(base_log_folder):
        os.makedirs(base_log_folder)
    
    # Create run-specific folders
    run_log_folder = os.path.join(base_log_folder, run_name)
    
    if not os.path.exists(run_log_folder):
        os.makedirs(run_log_folder)
        logger.info(f"Created folder: {run_log_folder}")
    
    return run_log_folder, run_name

def log_predictions_to_wandb(gt_texts, pred_texts, prefix, epoch, max_samples=10):
    """Log prediction examples to WandB as a table."""
    data = []
    for i, (gt, pred) in enumerate(zip(gt_texts[:max_samples], pred_texts[:max_samples])):
        data.append([epoch, i, gt, pred])
    
    table = wandb.Table(
        columns=["Epoch", "Sample_ID", "Ground_Truth", "Prediction"],
        data=data
    )
    wandb.log({f"{prefix}_predictions": table})

def train_process(augmentation=False, epochs=EPOCHS, wandb_project="simple-esposalles-ocr", 
                 wandb_name=None, run_name=None):
    """
    Main training function adapted for simplified dataset - keeping same interface as original
    
    Args:
        augmentation: Whether to use augmented dataset
        epochs: Number of training epochs
        wandb_project: WandB project name
        wandb_name: WandB run name
        run_name: Local run name for folders
    """
    
    # Create run-specific folders
    log_folder, actual_run_name = create_run_folders(run_name)
    
    # If wandb_name is not provided, use the run_name
    if wandb_name is None:
        wandb_name = actual_run_name
    
    # Initialize WandB - UPDATED CONFIG
    config = {
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "epochs": epochs,
        "early_stop": EARLY_STOP,
        "augmentation": augmentation,
        "device": device,
        "architecture": "SimpleEsposallesTrOCR",
        "optimizer": "AdamW",  # Changed from Adam to AdamW
        "weight_decay": WEIGHT_DECAY,  # Added weight decay
        "scheduler": "StepLR",
        "step_size": STEP_SIZE,  # Added step size
        "run_name": actual_run_name,
        "log_folder": log_folder
    }
    
    wandb.init(
        project=wandb_project,
        name=wandb_name,
        config=config,
        resume=False
    )
    
    # Log folder paths to WandB
    logger.info(f"Run name: {actual_run_name}")
    logger.info(f"Log folder: {log_folder}")
    logger.info(f"Using augmentation: {augmentation}")
    
    # Initialize model and load datasets - SAME INTERFACE AS ORIGINAL
    model = SimpleEsposallesTrOCR()
    train_dataset, test_dataset = load_datasets(augmentation=augmentation)
    
    # Log dataset information
    wandb.log({
        "dataset/train_size": len(train_dataset),
        "dataset/test_size": len(test_dataset),
        "dataset/augmentation": augmentation
    })
    
    # Initialize data loaders - UPDATED BATCH SIZE
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        collate_fn=collate_fn, 
        shuffle=True, 
        num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        collate_fn=collate_fn, 
        shuffle=False, 
        num_workers=NUM_WORKERS
    )

    # Setup optimizer and learning rate scheduler - CHANGED TO ADAMW
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=0.9)

    # Move model to device
    model.to(device)
    
    # Watch model gradients and parameters
    wandb.watch(model, log="all", log_freq=100)

    # Initialize tracking variables - SAME AS ORIGINAL
    best_val_loss = float('inf')
    best_res = None
    best_epoch = 0
    early_stop_counter = 0

    # Training loop - SAME STRUCTURE AS ORIGINAL
    for epoch in tqdm(range(1, epochs + 1)):
        epoch_metrics = {"epoch": epoch}

        # Open log files for this epoch (in run-specific folder)
        log_tr = open(os.path.join(log_folder, f'train_{epoch}.log'), 'a', encoding='utf-8')
        log_te = open(os.path.join(log_folder, f'test_{epoch}.log'), 'a', encoding='utf-8')
        log_tr.write('GT | PREDICTION\n')
        log_te.write('GT | PREDICTION\n')
        
        # Training phase - ADDED GRADIENT CLIPPING
        model.train()
        total_train_loss = 0
        start_time = time.time()
        lr = scheduler.get_last_lr()[0]
        
        batch_count = 0
        for data in train_loader:
            if batch_count % 200 == 0:
                print(f"Processing batch: {batch_count}")
                
            images = data['images'].to(device)
            texts = data['texts']
            
            # Forward pass - ADAPTED FOR SIMPLE MODEL
            loss, logits = model(images, texts)
            
            # Backward pass and optimization - ADDED GRADIENT CLIPPING
            optimizer.zero_grad()
            loss.backward()
            
            # ADDED: Gradient clipping (same as first code)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            
            optimizer.step()
            
            total_train_loss += loss.item()
            batch_count += 1
            
            # Log gradient norm every 100 batches (same as first code)
            if batch_count % 100 == 0:
                logger.info(f"Batch {batch_count}, Loss: {loss.item():.4f}, Grad Norm: {grad_norm:.4f}")

        # Update learning rate - SAME AS ORIGINAL
        scheduler.step()
        
        # Calculate average losses - SAME AS ORIGINAL
        avg_train_loss = total_train_loss / len(train_loader)
        epoch_time = time.time() - start_time
        
        # Add training metrics to epoch metrics - SAME AS ORIGINAL
        epoch_metrics.update({
            "train/loss": avg_train_loss,
            "train/learning_rate": lr,
            "train/epoch_time": epoch_time
        })
        
        # Log training progress - SAME AS ORIGINAL
        logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, "
                    f"lr: {lr:.7f}, time: {epoch_time:.2f}s")

        # Evaluation every 2 epochs - SAME AS ORIGINAL
        if epoch % 1 == 0:
            # Evaluation on training set (using autoregressive generation)
            model.eval()
            pred_text_tot_train = []
            gt_text_tot_train = []
            
            with torch.no_grad():
                for data in train_loader:
                    images = data['images'].to(device)
                    texts = data['texts']
                    
                    # Use autoregressive generation (no teacher forcing) - SAME AS ORIGINAL
                    pred_text = model.predict(images)
                    
                    pred_text_tot_train.extend(pred_text)
                    gt_text_tot_train.extend(texts)
            
            # Calculate accuracy (exact string match) - SAME AS ORIGINAL
            correct_train = sum(1 for gt, pred in zip(gt_text_tot_train, pred_text_tot_train) if gt.strip().lower() == pred.strip().lower())
            accuracy_train = correct_train / len(gt_text_tot_train)
            
            # Calculate WER and CER for training set - SAME AS ORIGINAL
            wer_train = jiwer.wer(gt_text_tot_train, pred_text_tot_train)
            cer_train = jiwer.cer(gt_text_tot_train, pred_text_tot_train)
            
            # Add training evaluation metrics - SAME AS ORIGINAL
            epoch_metrics.update({
                "train/wer": wer_train * 100,
                "train/cer": cer_train * 100,
                "train/accuracy": accuracy_train * 100
            })
            
            # Log training set results - SAME AS ORIGINAL
            for gt, pred in zip(gt_text_tot_train, pred_text_tot_train):
                log_tr.write(f'{gt} | {pred}\n')
                
            logger.info(f'[TRAIN SET] Accuracy: {accuracy_train*100:.2f}%, CER: {cer_train*100:.2f}%, WER: {wer_train*100:.2f}%')

            # Evaluation on test set - SAME AS ORIGINAL
            pred_text_tot_test = []
            gt_text_tot_test = []
            
            with torch.no_grad():
                for data in test_loader:
                    images = data['images'].to(device)
                    texts = data['texts']
                    
                    # Use autoregressive generation (no teacher forcing) - SAME AS ORIGINAL
                    pred_text = model.predict(images)
                    
                    pred_text_tot_test.extend(pred_text)
                    gt_text_tot_test.extend(texts)
            
            # Calculate accuracy (exact string match) - SAME AS ORIGINAL
            correct_test = sum(1 for gt, pred in zip(gt_text_tot_test, pred_text_tot_test) if gt.strip().lower() == pred.strip().lower())
            accuracy_test = correct_test / len(gt_text_tot_test)
            
            # Calculate WER and CER for test set - SAME AS ORIGINAL
            wer_test = jiwer.wer(gt_text_tot_test, pred_text_tot_test)
            cer_test = jiwer.cer(gt_text_tot_test, pred_text_tot_test)
            
            # Use WER on test set as validation metric for early stopping - SAME AS ORIGINAL
            validation_metric = wer_test
            
            # Add test evaluation metrics - SAME AS ORIGINAL
            epoch_metrics.update({
                "test/wer": wer_test * 100,
                "test/cer": cer_test * 100,
                "test/accuracy": accuracy_test * 100,
                "validation_metric": validation_metric,
                "early_stop_counter": early_stop_counter
            })

            logger.info(f'[TEST SET] Accuracy: {accuracy_test*100:.2f}%, CER: {cer_test*100:.2f}%, WER: {wer_test*100:.2f}%')
            logger.info(f'[TRAIN vs TEST] Train: Acc {accuracy_train*100:.2f}%, CER {cer_train*100:.2f}%, WER {wer_train*100:.2f}% | Test: Acc {accuracy_test*100:.2f}%, CER {cer_test*100:.2f}%, WER {wer_test*100:.2f}%')

            # Log test set results to file - SAME AS ORIGINAL
            for gt, pred in zip(gt_text_tot_test, pred_text_tot_test):
                log_te.write(f'{gt} | {pred}\n')

            # Log prediction examples to WandB every 10 epochs - SAME AS ORIGINAL
            if epoch % 10 == 0:
                log_predictions_to_wandb(gt_text_tot_train, pred_text_tot_train, "train", epoch)
                log_predictions_to_wandb(gt_text_tot_test, pred_text_tot_test, "test", epoch)
        else:
            # For non-evaluation epochs, set dummy values - SAME AS ORIGINAL
            validation_metric = best_val_loss  # Keep previous best
            wer_test = float('inf')  # Dummy value
            cer_test = float('inf')  # Dummy value
            accuracy_test = 0.0  # Dummy value
            
            # Add placeholder metrics for non-eval epochs - SAME AS ORIGINAL
            epoch_metrics.update({
                "validation_metric": validation_metric,
                "early_stop_counter": early_stop_counter
            })

        # Early stopping check (only on evaluation epochs) - SAME AS ORIGINAL
        if epoch % 2 == 0:
            if validation_metric < best_val_loss:
                best_val_loss = validation_metric
                best_res = (f'Best Epoch {epoch}, train: Acc {accuracy_train*100:.2f}% CER {cer_train*100:.2f}% WER {wer_train*100:.2f}%, '
                            f'test: Acc {accuracy_test*100:.2f}% CER {cer_test*100:.2f}% WER {wer_test*100:.2f}%')
                early_stop_counter = 0
                best_epoch = epoch
                
                # Add best model metrics - SAME AS ORIGINAL
                epoch_metrics.update({
                    "best/epoch": best_epoch,
                    "best/wer": wer_test * 100,
                    "best/cer": cer_test * 100,
                    "best/accuracy": accuracy_test * 100,
                    "best/validation_metric": validation_metric
                })
                
            else:
                early_stop_counter += 1
                if early_stop_counter >= EARLY_STOP:
                    logger.info(f"Early stopping at Epoch {epoch}, best epoch is {best_epoch}")
                    epoch_metrics["early_stopped"] = True
                    wandb.log(epoch_metrics, step=epoch)
                    break
        
        # Log all epoch metrics to WandB once per epoch - SAME AS ORIGINAL
        wandb.log(epoch_metrics, step=epoch)
        
        # Close log files - SAME AS ORIGINAL
        log_tr.close()
        log_te.close()
    
    # Log final summary - SAME AS ORIGINAL
    wandb.log({
        "final/best_epoch": best_epoch,
        "final/best_validation_metric": best_val_loss,
        "final/total_epochs_trained": epoch,
        "final/run_name": actual_run_name,
        "final/augmentation": augmentation
    }, step=epoch)

    # Save a summary file with run information - UPDATED SUMMARY
    summary_path = os.path.join(log_folder, 'run_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Run Name: {actual_run_name}\n")
        f.write(f"WandB Project: {wandb_project}\n")
        f.write(f"WandB Name: {wandb_name}\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Best Validation Metric: {best_val_loss}\n")
        f.write(f"Total Epochs Trained: {epoch}\n")
        f.write(f"Best Result: {best_res}\n")
        f.write(f"Augmentation: {augmentation}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Learning Rate: {LR}\n")
        f.write(f"Weight Decay: {WEIGHT_DECAY}\n")
        f.write(f"Step Size: {STEP_SIZE}\n")
        f.write(f"Optimizer: AdamW\n")
        f.write(f"Log Folder: {log_folder}\n")

    # Finish WandB run - SAME AS ORIGINAL
    wandb.finish()
    
    return best_res, actual_run_name

if __name__ == '__main__':
    
    print("=== Training without augmentation ===")
    best_res_normal, run_name_normal = train_process(
        augmentation=False,
        epochs=50,
        wandb_project="trocr-BERT-Hand",
        wandb_name="SimpleEsposalles-AdamW-5e6-batch6-OneEoch",
        run_name="SimpleEsposalles-AdamW-5e6-OneEoch"
    )
    