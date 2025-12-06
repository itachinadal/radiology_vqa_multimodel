"""Training script for Medical VQA models"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from tqdm import tqdm
import os
from datetime import datetime

from config import Config
from utils import AverageMeter, set_seed, create_directories, save_checkpoint, save_results
from metrics import VQAMetrics, evaluate_model


class Trainer:
    """Trainer class for Medical VQA models"""
    
    def __init__(self, model, train_loader, val_loader, config, 
                 model_name="model", use_text_encoder=False):
        self.model = model.to(config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.model_name = model_name
        self.use_text_encoder = use_text_encoder
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.NUM_EPOCHS
        )
        
        # Create directories
        self.checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, model_name)
        self.results_dir = os.path.join(config.RESULTS_DIR, model_name)
        create_directories([self.checkpoint_dir, self.results_dir])
        
        # Tracking
        self.best_val_accuracy = 0.0
        self.train_history = []
        self.val_history = []
        
        # Tokenizer for ViT model
        if use_text_encoder:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                "emilyalsentzer/Bio_ClinicalBERT"
            )
        else:
            self.tokenizer = None
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        losses = AverageMeter()
        metrics = VQAMetrics()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.NUM_EPOCHS}")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.config.DEVICE)
            questions = batch['question']
            answers = batch['answer']
            
            # Prepare inputs based on model type
            if self.use_text_encoder:
                # For ViT-ClinicalBERT: tokenize text questions
                if isinstance(questions, list):
                    encoded = self.tokenizer(
                        questions,
                        padding=True,
                        truncation=True,
                        max_length=self.config.MAX_QUESTION_LENGTH,
                        return_tensors='pt'
                    )
                    questions = encoded['input_ids'].to(self.config.DEVICE)
                    attention_mask = encoded['attention_mask'].to(self.config.DEVICE)
                else:
                    attention_mask = None
                
                # Forward pass
                outputs = self.model(images, questions, attention_mask, mode='classify')
            else:
                # For baseline: questions are already tokenized
                if not isinstance(questions, torch.Tensor):
                    continue
                questions = questions.to(self.config.DEVICE)
                outputs = self.model(images, questions, mode='classify')
            
            # Prepare targets
            if isinstance(answers, torch.Tensor):
                targets = answers.to(self.config.DEVICE)
            else:
                # Simple answer mapping for text answers
                answer_to_id = {
                    'yes': 0, 'no': 1, 'normal': 2, 'abnormal': 3,
                    'mild abnormality': 4, 'moderate abnormality': 5,
                    'severe abnormality': 6
                }
                targets = torch.tensor([
                    answer_to_id.get(ans.lower(), 0) for ans in answers
                ]).to(self.config.DEVICE)
            
            # Compute loss
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update metrics
            losses.update(loss.item(), images.size(0))
            predictions = torch.argmax(outputs, dim=1)
            metrics.update(predictions, targets)
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{losses.avg:.4f}'})
        
        # Compute metrics
        epoch_metrics = metrics.compute_classification_metrics()
        epoch_metrics['loss'] = losses.avg
        
        return epoch_metrics
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        
        losses = AverageMeter()
        metrics = VQAMetrics()
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                images = batch['image'].to(self.config.DEVICE)
                questions = batch['question']
                answers = batch['answer']
                
                # Prepare inputs
                if self.use_text_encoder:
                    if isinstance(questions, list):
                        encoded = self.tokenizer(
                            questions,
                            padding=True,
                            truncation=True,
                            max_length=self.config.MAX_QUESTION_LENGTH,
                            return_tensors='pt'
                        )
                        questions = encoded['input_ids'].to(self.config.DEVICE)
                        attention_mask = encoded['attention_mask'].to(self.config.DEVICE)
                    else:
                        attention_mask = None
                    
                    outputs = self.model(images, questions, attention_mask, mode='classify')
                else:
                    if not isinstance(questions, torch.Tensor):
                        continue
                    questions = questions.to(self.config.DEVICE)
                    outputs = self.model(images, questions, mode='classify')
                
                # Prepare targets
                if isinstance(answers, torch.Tensor):
                    targets = answers.to(self.config.DEVICE)
                else:
                    answer_to_id = {
                        'yes': 0, 'no': 1, 'normal': 2, 'abnormal': 3,
                        'mild abnormality': 4, 'moderate abnormality': 5,
                        'severe abnormality': 6
                    }
                    targets = torch.tensor([
                        answer_to_id.get(ans.lower(), 0) for ans in answers
                    ]).to(self.config.DEVICE)
                
                # Compute loss
                loss = self.criterion(outputs, targets)
                losses.update(loss.item(), images.size(0))
                
                # Update metrics
                predictions = torch.argmax(outputs, dim=1)
                metrics.update(predictions, targets)
        
        # Compute metrics
        val_metrics = metrics.compute_classification_metrics()
        val_metrics['loss'] = losses.avg
        
        return val_metrics
    
    def train(self):
        """Full training loop"""
        print(f"\nTraining {self.model_name}...")
        print(f"Device: {self.config.DEVICE}")
        print(f"Epochs: {self.config.NUM_EPOCHS}")
        print(f"Batch size: {self.config.BATCH_SIZE}")
        print(f"Learning rate: {self.config.LEARNING_RATE}\n")
        
        for epoch in range(self.config.NUM_EPOCHS):
            # Train
            train_metrics = self.train_epoch(epoch)
            self.train_history.append(train_metrics)
            
            # Validate
            val_metrics = self.validate()
            self.val_history.append(val_metrics)
            
            # Update learning rate
            self.scheduler.step()
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.config.NUM_EPOCHS}")
            print(f"Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Acc: {train_metrics['accuracy']:.4f}, "
                  f"F1: {train_metrics['f1']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Acc: {val_metrics['accuracy']:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}")
            
            # Save best model
            if val_metrics['accuracy'] > self.best_val_accuracy:
                self.best_val_accuracy = val_metrics['accuracy']
                checkpoint_path = os.path.join(
                    self.checkpoint_dir,
                    f'{self.model_name}_best.pth'
                )
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_metrics,
                    checkpoint_path
                )
                print(f"New best model saved! Accuracy: {self.best_val_accuracy:.4f}")
            
            # Save periodic checkpoint
            if (epoch + 1) % self.config.SAVE_EVERY_N_EPOCHS == 0:
                checkpoint_path = os.path.join(
                    self.checkpoint_dir,
                    f'{self.model_name}_epoch_{epoch+1}.pth'
                )
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_metrics,
                    checkpoint_path
                )
        
        # Save training history
        history = {
            'train': self.train_history,
            'val': self.val_history,
            'best_val_accuracy': self.best_val_accuracy
        }
        save_results(history, os.path.join(self.results_dir, 'training_history.json'))
        
        print(f"\nTraining completed!")
        print(f"Best validation accuracy: {self.best_val_accuracy:.4f}")
        
        return history


if __name__ == "__main__":
    from config import Config
    from dataset import create_dataloaders
    from baseline_model import BaselineCNNLSTM
    
    # Setup
    config = Config()
    set_seed(config.SEED)
    
    # Create dataloaders
    train_loader, val_loader, test_loader, vocab = create_dataloaders(
        config, use_synthetic=True
    )
    
    # Create model
    model = BaselineCNNLSTM(
        vocab_size=len(vocab),
        lstm_hidden_dim=config.LSTM_HIDDEN_DIM,
        lstm_num_layers=config.LSTM_NUM_LAYERS,
        num_answers=10
    )
    
    # Create trainer
    trainer = Trainer(
        model, train_loader, val_loader, config,
        model_name="baseline_test",
        use_text_encoder=False
    )
    
    # Test one epoch
    config.NUM_EPOCHS = 2
    history = trainer.train()
