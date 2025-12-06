"""Configuration file for Medical VQA experiments"""

import torch

class Config:
    # Model selection
    MODEL_TYPE = "vit"  # Options: "baseline" (CNN-LSTM) or "vit" (ViT-ClinicalBERT)
    
    # Paths
    DATA_DIR = "data"
    CHECKPOINT_DIR = "checkpoints"
    RESULTS_DIR = "results"
    
    # Image settings
    IMAGE_SIZE = 224
    PATCH_SIZE = 16
    NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
    
    # Model hyperparameters - Baseline (CNN-LSTM)
    CNN_FEATURE_DIM = 2048  # ResNet50 output
    LSTM_HIDDEN_DIM = 512
    LSTM_NUM_LAYERS = 2
    
    # Model hyperparameters - Proposed (ViT + ClinicalBERT)
    VIT_MODEL = "google/vit-base-patch16-224"
    VIT_HIDDEN_DIM = 768
    TEXT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
    TEXT_HIDDEN_DIM = 768
    
    # Fusion and decoder
    FUSION_DIM = 512
    NUM_ATTENTION_HEADS = 8
    NUM_DECODER_LAYERS = 2
    DROPOUT = 0.1
    
    # Vocabulary
    MAX_QUESTION_LENGTH = 50
    MAX_ANSWER_LENGTH = 30
    
    # Training hyperparameters
    BATCH_SIZE = 16
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01
    WARMUP_STEPS = 500
    
    # Answer types
    ANSWER_TYPES = ["yes/no", "choice", "freeform"]
    MAX_CHOICES = 4
    
    # Evaluation
    EVAL_EVERY_N_STEPS = 500
    SAVE_EVERY_N_EPOCHS = 5
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4
    
    # Random seed
    SEED = 42
