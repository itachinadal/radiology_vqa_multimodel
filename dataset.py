"""Dataset classes for Medical VQA"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import json
import os
import random
from typing import Dict, List, Tuple

class MedicalVQADataset(Dataset):
    """Medical VQA Dataset"""
    
    def __init__(self, data_path: str, image_dir: str, transform=None, 
                 max_question_len: int = 50, max_answer_len: int = 30,
                 vocab: Dict = None, is_train: bool = True):
        """
        Args:
            data_path: Path to JSON file with QA pairs
            image_dir: Directory containing medical images
            transform: Image transformations
            max_question_len: Maximum question length
            max_answer_len: Maximum answer length
            vocab: Vocabulary dictionary
            is_train: Whether this is training data
        """
        self.image_dir = image_dir
        self.transform = transform
        self.max_question_len = max_question_len
        self.max_answer_len = max_answer_len
        self.vocab = vocab
        self.is_train = is_train
        
        # Load data
        with open(data_path, 'r') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load and transform image
        image_path = os.path.join(self.image_dir, item['image'])
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except:
            # Return a blank image if loading fails
            image = torch.zeros(3, 224, 224)
        
        # Get question and answer
        question = item['question']
        answer = item['answer']
        answer_type = item.get('answer_type', 'freeform')
        
        # Tokenize if using simple vocab (for baseline)
        if self.vocab is not None:
            question_tokens = self._tokenize(question, self.max_question_len)
            answer_tokens = self._tokenize(answer, self.max_answer_len)
        else:
            # For transformer models, return raw text
            question_tokens = question
            answer_tokens = answer
        
        return {
            'image': image,
            'question': question_tokens,
            'answer': answer_tokens,
            'answer_type': answer_type,
            'raw_question': question,
            'raw_answer': answer
        }
    
    def _tokenize(self, text: str, max_len: int):
        """Simple tokenization using vocabulary"""
        words = text.lower().split()
        tokens = [self.vocab.get(word, self.vocab.get('<UNK>', 1)) for word in words]
        
        # Pad or truncate
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        else:
            tokens = tokens + [self.vocab.get('<PAD>', 0)] * (max_len - len(tokens))
        
        return torch.LongTensor(tokens)


class SyntheticMedicalVQADataset(Dataset):
    """Synthetic dataset for testing (when real data is not available)"""
    
    def __init__(self, num_samples: int = 1000, image_size: int = 224, 
                 transform=None, vocab: Dict = None):
        self.num_samples = num_samples
        self.image_size = image_size
        self.transform = transform
        self.vocab = vocab
        
        # Question templates
        self.question_templates = [
            "Is there evidence of {} in the {}?",
            "What is the condition of the {}?",
            "Are there any abnormalities in the {}?",
            "Is the {} normal?",
            "What pathology is present in the {}?"
        ]
        
        # Medical terms
        self.pathologies = ["pneumonia", "effusion", "consolidation", "edema", "nodule"]
        self.anatomies = ["left lung", "right lung", "heart", "chest", "thorax"]
        self.answers_binary = ["yes", "no"]
        self.answers_descriptive = ["normal", "abnormal", "mild abnormality", 
                                    "moderate abnormality", "severe abnormality"]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate synthetic image (random noise, representing X-ray)
        image = torch.randn(3, self.image_size, self.image_size)
        if self.transform:
            image = self.transform(image)
        
        # Generate question
        template = random.choice(self.question_templates)
        if "{}" in template:
            if template.count("{}") == 2:
                question = template.format(random.choice(self.pathologies), 
                                         random.choice(self.anatomies))
            else:
                question = template.format(random.choice(self.anatomies))
        else:
            question = template
        
        # Generate answer based on question
        if "Is there" in question or "normal" in question:
            answer = random.choice(self.answers_binary)
            answer_type = "yes/no"
        else:
            answer = random.choice(self.answers_descriptive)
            answer_type = "freeform"
        
        # Tokenize if using vocab
        if self.vocab is not None:
            question_tokens = self._tokenize(question, 50)
            answer_tokens = self._tokenize(answer, 30)
        else:
            question_tokens = question
            answer_tokens = answer
        
        return {
            'image': image,
            'question': question_tokens,
            'answer': answer_tokens,
            'answer_type': answer_type,
            'raw_question': question,
            'raw_answer': answer
        }
    
    def _tokenize(self, text: str, max_len: int):
        """Simple tokenization"""
        words = text.lower().split()
        tokens = [self.vocab.get(word, self.vocab.get('<UNK>', 1)) for word in words]
        
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        else:
            tokens = tokens + [self.vocab.get('<PAD>', 0)] * (max_len - len(tokens))
        
        return torch.LongTensor(tokens)


def get_transforms(image_size: int = 224, is_train: bool = True):
    """Get image transformations with augmentation"""
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


def create_dataloaders(config, use_synthetic: bool = True):
    """Create train, validation, and test dataloaders"""
    
    if use_synthetic:
        # Use synthetic data for testing
        print("Using synthetic dataset for demonstration...")
        
        # Build simple vocabulary
        vocab = {
            '<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3,
            'is': 4, 'there': 5, 'evidence': 6, 'of': 7, 'in': 8, 'the': 9,
            'what': 10, 'condition': 11, 'are': 12, 'any': 13, 'abnormalities': 14,
            'normal': 15, 'pathology': 16, 'present': 17, 'yes': 18, 'no': 19,
            'pneumonia': 20, 'effusion': 21, 'consolidation': 22, 'edema': 23,
            'nodule': 24, 'left': 25, 'lung': 26, 'right': 27, 'heart': 28,
            'chest': 29, 'thorax': 30, 'abnormal': 31, 'mild': 32, 'abnormality': 33,
            'moderate': 34, 'severe': 35
        }
        
        train_dataset = SyntheticMedicalVQADataset(
            num_samples=800,
            image_size=config.IMAGE_SIZE,
            transform=get_transforms(config.IMAGE_SIZE, is_train=True),
            vocab=None if config.MODEL_TYPE == "vit" else vocab
        )
        
        val_dataset = SyntheticMedicalVQADataset(
            num_samples=100,
            image_size=config.IMAGE_SIZE,
            transform=get_transforms(config.IMAGE_SIZE, is_train=False),
            vocab=None if config.MODEL_TYPE == "vit" else vocab
        )
        
        test_dataset = SyntheticMedicalVQADataset(
            num_samples=100,
            image_size=config.IMAGE_SIZE,
            transform=get_transforms(config.IMAGE_SIZE, is_train=False),
            vocab=None if config.MODEL_TYPE == "vit" else vocab
        )
    else:
        # Load real datasets (SLAKE, IUXRAY, etc.)
        # This would be implemented when real data is available
        raise NotImplementedError("Real dataset loading not implemented yet")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, vocab if use_synthetic else None
