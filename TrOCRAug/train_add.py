import torch
import time
import jiwer
import os
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from simple_dataset import load_simple_data, simple_collate_fn, load_simple_data_probabilistic
from model_add import ImageTextTrOCRModified
import logging
import wandb
from datetime import datetime

# Set CUDA device (can be overridden via environment variable)
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Choose device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device: {device}")

# Simple training hyperparameters
BATCH_SIZE = 2
LR = 5e-6
EPOCHS = 10
NUM_WORKERS = 2
STEP_SIZE = 5
WEIGHT_DECAY = 1e-2

def create_log_folder(run_name):
    """
    Create organized log folder structure: logs/{run_name}/
    """
    base_logs_dir = "logs"
    run_log_dir = os.path.join(base_logs_dir, run_name)
    
    # Create directories if they don't exist
    os.makedirs(run_log_dir, exist_ok=True)
    
    logger.info(f"Created log directory: {run_log_dir}")
    return run_log_dir

def simple_train_probabilistic(category_percentages=None, epochs=EPOCHS, wandb_project="wordart-probabilistic", run_name=None):
    """
    NEW: Probabilistic training function for WordArt where recognition is always included 
    and exactly one of 4 big categories is selected based on percentages that sum to 100%
    """
    
    # Default percentages if none provided
    if category_percentages is None:
        category_percentages = {
            "CHARACTER_PRESENCE": 25,
            "POSITIONAL_ANALYSIS": 25,
            "STRUCTURAL_ANALYSIS": 25,
            "BOUNDARY_ANALYSIS": 25
        }
    
    # Validate percentages sum to 100
    total_percentage = sum(category_percentages.values())
    if abs(total_percentage - 100) > 0.01:
        raise ValueError(f"Category percentages must sum to 100, but got {total_percentage}")
    
    # Initialize WandB
    if run_name is None:
        run_name = f"wordart_prob_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create log folder for this specific run
    run_log_dir = create_log_folder(run_name)
    
    wandb.init(
        project=wandb_project,
        name=run_name,
        config={
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "epochs": epochs,
            "category_percentages": category_percentages,
            "device": device,
            "log_directory": run_log_dir,
            "step_size": STEP_SIZE,
            "weight_decay": WEIGHT_DECAY,
            "training_mode": "probabilistic_percentage_based",
            "dataset": "WordArt"
        }
    )
    
    # Load probabilistic datasets
    logger.info("Loading probabilistic WordArt datasets...")
    data_train_prob, data_test_prob, data_train_rec, data_test_rec = load_simple_data_probabilistic(
        category_percentages=category_percentages
    )
    
    logger.info(f"Train size: {len(data_train_prob)}, Test size: {len(data_test_prob)}")
    logger.info(f"Category percentages: {category_percentages}")
    
    # Log dataset sizes to WandB
    wandb.log({
        "dataset/train_size": len(data_train_prob),
        "dataset/test_size": len(data_test_prob)
    })
    
    # Create data loaders
    train_loader = DataLoader(
        data_train_prob, 
        batch_size=BATCH_SIZE, 
        collate_fn=simple_collate_fn, 
        shuffle=True, 
        num_workers=NUM_WORKERS
    )
    
    rec_train_loader = DataLoader(
        data_train_rec, 
        batch_size=BATCH_SIZE, 
        collate_fn=simple_collate_fn, 
        shuffle=False, 
        num_workers=NUM_WORKERS
    )
    
    rec_test_loader = DataLoader(
        data_test_rec, 
        batch_size=BATCH_SIZE, 
        collate_fn=simple_collate_fn, 
        shuffle=False, 
        num_workers=NUM_WORKERS
    )
    
    # Initialize model
    logger.info("Initializing model...")
    model = ImageTextTrOCRModified()
    model.to(device)
    
    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=0.9)
    
    best_wer = float('inf')
    best_epoch = 0
    
    # Training loop
    for epoch in range(1, epochs + 1):
        logger.info(f"Starting Epoch {epoch}/{epochs}")
        
        # Training phase
        model.train()
        total_train_loss = 0
        batch_count = 0
        
        start_time = time.time()
        
        for batch_data in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
            pixel_values = batch_data['pixel_values'].to(device)
            questions = batch_data['questions']
            answers = batch_data['answers']
            
            # Forward pass
            loss, logits = model(pixel_values, answers, questions)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            batch_count += 1
            
            if batch_count % 100 == 0:
                logger.info(f"Batch {batch_count}, Loss: {loss.item():.4f}, Grad Norm: {grad_norm:.4f}")
        
        # Update learning rate
        scheduler.step()
        
        avg_train_loss = total_train_loss / len(train_loader)
        epoch_time = time.time() - start_time
        
        # Log training metrics
        wandb.log({
            "epoch": epoch,
            "train/loss": avg_train_loss,
            "train/learning_rate": scheduler.get_last_lr()[0],
            "train/epoch_time": epoch_time,
        })
        
        logger.info(f"Epoch {epoch} Training - Loss: {avg_train_loss:.4f}, Time: {epoch_time:.2f}s")
        
        # Evaluation phase (every 2 epochs)
        if epoch % 2 == 0:
            model.eval()
            
            # Evaluate on training set
            train_pred_texts = []
            train_gt_texts = []
            
            logger.info("Evaluating on training set...")
            with torch.no_grad():
                for batch_data in tqdm(rec_train_loader, desc="Eval Train"):
                    pixel_values = batch_data['pixel_values'].to(device)
                    questions = batch_data['questions']
                    gt_answers = batch_data['answers']
                    
                    predictions = model.predict(pixel_values, questions)
                    train_pred_texts.extend(predictions)
                    train_gt_texts.extend(gt_answers)
            
            # Evaluate on test set
            test_pred_texts = []
            test_gt_texts = []
            
            logger.info("Evaluating on test set...")
            with torch.no_grad():
                for batch_data in tqdm(rec_test_loader, desc="Eval Test"):
                    pixel_values = batch_data['pixel_values'].to(device)
                    questions = batch_data['questions']
                    gt_answers = batch_data['answers']
                    
                    predictions = model.predict(pixel_values, questions)
                    test_pred_texts.extend(predictions)
                    test_gt_texts.extend(gt_answers)
            
            # Write prediction logs to run-specific folder
            train_log_file = os.path.join(run_log_dir, f'train_predictions_epoch_{epoch}.log')
            test_log_file = os.path.join(run_log_dir, f'test_predictions_epoch_{epoch}.log')
            
            with open(train_log_file, 'w', encoding='utf-8') as f:
                f.write('GT | PREDICTION\n')
                for gt, pred in zip(train_gt_texts, train_pred_texts):
                    f.write(f'{gt} | {pred}\n')
            
            with open(test_log_file, 'w', encoding='utf-8') as f:
                f.write('GT | PREDICTION\n')
                for gt, pred in zip(test_gt_texts, test_pred_texts):
                    f.write(f'{gt} | {pred}\n')
            
            logger.info(f"Saved predictions to {train_log_file} and {test_log_file}")
            
            # Calculate metrics for both sets
            train_wer = jiwer.wer(train_gt_texts, train_pred_texts)
            train_cer = jiwer.cer(train_gt_texts, train_pred_texts)
            test_wer = jiwer.wer(test_gt_texts, test_pred_texts)
            test_cer = jiwer.cer(test_gt_texts, test_pred_texts)
            
            # Log evaluation metrics
            wandb.log({
                "epoch": epoch,
                "train/wer": train_wer * 100,
                "train/cer": train_cer * 100,
                "test/wer": test_wer * 100,
                "test/cer": test_cer * 100
            })
            
            logger.info(f"Epoch {epoch} Evaluation:")
            logger.info(f"  Train - WER: {train_wer*100:.2f}%, CER: {train_cer*100:.2f}%")
            logger.info(f"  Test  - WER: {test_wer*100:.2f}%, CER: {test_cer*100:.2f}%")
            
            # Save best model based on test WER
            if test_wer < best_wer:
                best_wer = test_wer
                best_epoch = epoch
                logger.info(f"New best test WER: {best_wer*100:.2f}% at epoch {epoch}")
                
                # Log best metrics
                wandb.log({
                    "best/epoch": best_epoch,
                    "best/test_wer": best_wer * 100,
                    "best/test_cer": test_cer * 100
                })
            
            # Log sample predictions every 4 epochs
            if epoch % 4 == 0:
                logger.info("Sample predictions (Test set):")
                for i in range(min(3, len(test_gt_texts))):
                    logger.info(f"  GT: '{test_gt_texts[i]}' | Pred: '{test_pred_texts[i]}'")
    
    # Save final summary to run folder
    summary_file = os.path.join(run_log_dir, 'run_summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"Run Name: {run_name}\n")
        f.write(f"Category Percentages: {category_percentages}\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Best Test WER: {best_wer*100:.2f}%\n")
        f.write(f"Total Epochs: {epochs}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Learning Rate: {LR}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Log Directory: {run_log_dir}\n")
        f.write(f"Training Mode: Probabilistic Percentage-Based\n")
        f.write(f"Dataset: WordArt\n")
    
    logger.info(f"Saved run summary to {summary_file}")
    logger.info(f"Training completed. Best test WER: {best_wer*100:.2f}% at epoch {best_epoch}")
    
    # Finish WandB run
    wandb.finish()
    
    return best_wer, best_epoch


if __name__ == '__main__':
    category_percentages = {
        "CHARACTER_PRESENCE": 30,
        "POSITIONAL_ANALYSIS": 30,
        "STRUCTURAL_ANALYSIS": 25,
        "BOUNDARY_ANALYSIS": 15
    }
    best_wer, best_epoch = simple_train_probabilistic(
        category_percentages=category_percentages,
        epochs=50,
        wandb_project="BERT-Art-Probabilistic",
        run_name="30-30-25-15"
    )