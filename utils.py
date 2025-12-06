"""Utility functions for Medical VQA"""

import torch
import numpy as np
import random
import os
from typing import Dict, List
import json

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_directories(dirs: List[str]):
    """Create directories if they don't exist"""
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def save_checkpoint(model, optimizer, epoch, metrics, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['metrics']

def save_results(results: Dict, filepath: str):
    """Save results to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {filepath}")

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def build_vocab_from_data(questions: List[str], answers: List[str], min_freq: int = 2):
    """Build vocabulary from questions and answers"""
    from collections import Counter
    
    word_counter = Counter()
    for text in questions + answers:
        words = text.lower().split()
        word_counter.update(words)
    
    # Special tokens
    vocab = {
        '<PAD>': 0,
        '<UNK>': 1,
        '<SOS>': 2,
        '<EOS>': 3
    }
    
    # Add words that meet minimum frequency
    idx = len(vocab)
    for word, freq in word_counter.items():
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1
    
    return vocab

def tokenize_text(text: str, vocab: Dict[str, int], max_length: int):
    """Tokenize text using vocabulary"""
    words = text.lower().split()
    tokens = [vocab.get(word, vocab['<UNK>']) for word in words]
    
    # Truncate or pad
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    else:
        tokens = tokens + [vocab['<PAD>']] * (max_length - len(tokens))
    
    return tokens
