"""Evaluation metrics for Medical VQA"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from collections import Counter
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

class VQAMetrics:
    """Comprehensive metrics for VQA evaluation"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.predictions = []
        self.targets = []
        self.pred_texts = []
        self.target_texts = []
    
    def update(self, predictions, targets, pred_texts=None, target_texts=None):
        """
        Update metrics with new batch
        
        Args:
            predictions: tensor of predicted class indices
            targets: tensor of target class indices
            pred_texts: list of predicted text answers (for open-ended)
            target_texts: list of target text answers (for open-ended)
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        self.predictions.extend(predictions.tolist())
        self.targets.extend(targets.tolist())
        
        if pred_texts is not None and target_texts is not None:
            self.pred_texts.extend(pred_texts)
            self.target_texts.extend(target_texts)
    
    def compute_classification_metrics(self):
        """Compute classification metrics (accuracy, F1, precision, recall)"""
        if len(self.predictions) == 0:
            return {}
        
        accuracy = accuracy_score(self.targets, self.predictions)
        f1 = f1_score(self.targets, self.predictions, average='weighted', zero_division=0)
        precision = precision_score(self.targets, self.predictions, average='weighted', zero_division=0)
        recall = recall_score(self.targets, self.predictions, average='weighted', zero_division=0)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def compute_text_metrics(self):
        """Compute text generation metrics (BLEU, METEOR, exact match)"""
        if len(self.pred_texts) == 0:
            return {}
        
        bleu_scores = []
        meteor_scores = []
        exact_matches = 0
        
        smoothing = SmoothingFunction().method1
        
        for pred, target in zip(self.pred_texts, self.target_texts):
            # Tokenize
            pred_tokens = pred.lower().split()
            target_tokens = target.lower().split()
            
            # BLEU score
            try:
                bleu = sentence_bleu([target_tokens], pred_tokens, 
                                    smoothing_function=smoothing)
                bleu_scores.append(bleu)
            except:
                bleu_scores.append(0.0)
            
            # METEOR score
            try:
                meteor = meteor_score([target_tokens], pred_tokens)
                meteor_scores.append(meteor)
            except:
                meteor_scores.append(0.0)
            
            # Exact match
            if pred.lower().strip() == target.lower().strip():
                exact_matches += 1
        
        return {
            'bleu': np.mean(bleu_scores) if bleu_scores else 0.0,
            'meteor': np.mean(meteor_scores) if meteor_scores else 0.0,
            'exact_match': exact_matches / len(self.pred_texts) if self.pred_texts else 0.0
        }
    
    def compute_all_metrics(self):
        """Compute all metrics"""
        metrics = {}
        
        # Classification metrics
        if len(self.predictions) > 0:
            metrics.update(self.compute_classification_metrics())
        
        # Text generation metrics
        if len(self.pred_texts) > 0:
            metrics.update(self.compute_text_metrics())
        
        return metrics


def evaluate_model(model, dataloader, device, vocab=None, id2answer=None, use_text=False):
    """
    Evaluate model on a dataset
    
    Args:
        model: VQA model
        dataloader: DataLoader for evaluation
        device: torch device
        vocab: vocabulary dict (for baseline)
        id2answer: mapping from answer IDs to text
        use_text: whether to use text encoder (ViT model)
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    metrics = VQAMetrics()
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            questions = batch['question']
            answers = batch['answer']
            
            if use_text:
                # For ViT-ClinicalBERT: tokenize questions
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
                
                # Handle raw text questions
                if isinstance(questions, list):
                    encoded = tokenizer(questions, padding=True, truncation=True, 
                                      max_length=50, return_tensors='pt')
                    questions = encoded['input_ids'].to(device)
                    attention_mask = encoded['attention_mask'].to(device)
                else:
                    attention_mask = None
                
                # Forward pass
                outputs = model(images, questions, attention_mask, mode='classify')
            else:
                # For baseline: questions are already tokenized
                if not isinstance(questions, torch.Tensor):
                    continue
                questions = questions.to(device)
                outputs = model(images, questions, mode='classify')
            
            # Get predictions
            predictions = torch.argmax(outputs, dim=1)
            
            # Convert answers to class indices (simplified)
            if isinstance(answers, torch.Tensor):
                targets = answers.to(device)
            else:
                # For text answers, create simple mapping
                answer_to_id = {'yes': 0, 'no': 1, 'normal': 2, 'abnormal': 3}
                targets = torch.tensor([answer_to_id.get(ans.lower(), 0) 
                                       for ans in answers]).to(device)
            
            # Update metrics
            metrics.update(predictions, targets)
    
    return metrics.compute_all_metrics()


if __name__ == "__main__":
    # Test metrics
    metrics = VQAMetrics()
    
    # Simulate some predictions
    predictions = torch.tensor([0, 1, 2, 3, 0, 1])
    targets = torch.tensor([0, 1, 2, 2, 0, 0])
    
    pred_texts = ["yes", "no", "normal", "abnormal", "yes", "yes"]
    target_texts = ["yes", "no", "normal", "normal", "yes", "no"]
    
    metrics.update(predictions, targets, pred_texts, target_texts)
    
    results = metrics.compute_all_metrics()
    print("Metrics:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")
