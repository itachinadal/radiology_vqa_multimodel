"""Main experiment runner for Medical VQA"""

import torch
import argparse
import os
from datetime import datetime

from config import Config
from utils import set_seed, create_directories
from dataset import create_dataloaders
from baseline_model import BaselineCNNLSTM
from proposed_model import ViTClinicalBERTVQA
from train import Trainer
from metrics import evaluate_model


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_baseline_experiment(config, train_loader, val_loader, test_loader, vocab):
    """Run baseline CNN-LSTM experiment"""
    print("\n" + "="*80)
    print("BASELINE MODEL: CNN-LSTM")
    print("="*80)
    
    # Create model
    model = BaselineCNNLSTM(
        vocab_size=len(vocab),
        embedding_dim=300,
        lstm_hidden_dim=config.LSTM_HIDDEN_DIM,
        lstm_num_layers=config.LSTM_NUM_LAYERS,
        cnn_feature_dim=config.CNN_FEATURE_DIM,
        fusion_dim=config.FUSION_DIM,
        num_answers=10,  # Simplified: yes, no, normal, abnormal, etc.
        dropout=config.DROPOUT
    )
    
    print(f"\nModel parameters: {count_parameters(model):,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        model_name="baseline_cnn_lstm",
        use_text_encoder=False
    )
    
    # Train
    history = trainer.train()
    
    # Evaluate on test set
    print("\n" + "-"*80)
    print("Testing baseline model...")
    test_metrics = evaluate_model(
        model, test_loader, config.DEVICE,
        vocab=vocab, use_text=False
    )
    
    print("\nTest Results:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    return {
        'model_name': 'baseline_cnn_lstm',
        'parameters': count_parameters(model),
        'best_val_accuracy': history['best_val_accuracy'],
        'test_metrics': test_metrics,
        'history': history
    }


def run_proposed_experiment(config, train_loader, val_loader, test_loader):
    """Run proposed ViT-ClinicalBERT experiment"""
    print("\n" + "="*80)
    print("PROPOSED MODEL: ViT-ClinicalBERT")
    print("="*80)
    
    # Create model
    model = ViTClinicalBERTVQA(
        vit_model_name=config.VIT_MODEL,
        text_model_name=config.TEXT_MODEL,
        fusion_dim=config.FUSION_DIM,
        num_attention_heads=config.NUM_ATTENTION_HEADS,
        num_decoder_layers=config.NUM_DECODER_LAYERS,
        num_answers=10,
        dropout=config.DROPOUT
    )
    
    param_counts = {
        'total': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'vit': sum(p.numel() for p in model.vit.parameters() if p.requires_grad),
        'text': sum(p.numel() for p in model.text_encoder.parameters() if p.requires_grad)
    }
    param_counts['fusion'] = param_counts['total'] - param_counts['vit'] - param_counts['text']
    
    print(f"\nModel parameters:")
    print(f"  Total: {param_counts['total']:,}")
    print(f"  ViT: {param_counts['vit']:,}")
    print(f"  ClinicalBERT: {param_counts['text']:,}")
    print(f"  Fusion+Decoder: {param_counts['fusion']:,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        model_name="vit_clinicalbert",
        use_text_encoder=True
    )
    
    # Train
    history = trainer.train()
    
    # Evaluate on test set
    print("\n" + "-"*80)
    print("Testing proposed model...")
    test_metrics = evaluate_model(
        model, test_loader, config.DEVICE,
        use_text=True
    )
    
    print("\nTest Results:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    return {
        'model_name': 'vit_clinicalbert',
        'parameters': param_counts,
        'best_val_accuracy': history['best_val_accuracy'],
        'test_metrics': test_metrics,
        'history': history
    }


def compare_results(baseline_results, proposed_results):
    """Compare baseline and proposed model results"""
    print("\n" + "="*80)
    print("COMPARATIVE RESULTS")
    print("="*80)
    
    print("\n1. Model Size:")
    print(f"   Baseline:  {baseline_results['parameters']:,} parameters")
    if isinstance(proposed_results['parameters'], dict):
        print(f"   Proposed:  {proposed_results['parameters']['total']:,} parameters")
    else:
        print(f"   Proposed:  {proposed_results['parameters']:,} parameters")
    
    print("\n2. Best Validation Accuracy:")
    baseline_val = baseline_results['best_val_accuracy']
    proposed_val = proposed_results['best_val_accuracy']
    improvement = (proposed_val - baseline_val) * 100
    print(f"   Baseline:  {baseline_val:.4f} ({baseline_val*100:.2f}%)")
    print(f"   Proposed:  {proposed_val:.4f} ({proposed_val*100:.2f}%)")
    print(f"   Improvement: {improvement:+.2f} percentage points")
    
    print("\n3. Test Set Performance:")
    print(f"   {'Metric':<15} {'Baseline':<12} {'Proposed':<12} {'Improvement':<12}")
    print(f"   {'-'*15} {'-'*12} {'-'*12} {'-'*12}")
    
    for metric in ['accuracy', 'f1', 'precision', 'recall']:
        if metric in baseline_results['test_metrics'] and metric in proposed_results['test_metrics']:
            baseline_val = baseline_results['test_metrics'][metric]
            proposed_val = proposed_results['test_metrics'][metric]
            improvement = (proposed_val - baseline_val) * 100
            print(f"   {metric:<15} {baseline_val:<12.4f} {proposed_val:<12.4f} {improvement:+.2f}%")
    
    # Save comparison
    comparison = {
        'baseline': baseline_results,
        'proposed': proposed_results,
        'timestamp': datetime.now().isoformat()
    }
    
    from utils import save_results
    save_results(comparison, os.path.join(Config.RESULTS_DIR, 'comparison.json'))


def main():
    """Main experiment runner"""
    parser = argparse.ArgumentParser(description='Medical VQA Experiments')
    parser.add_argument('--model', type=str, default='both',
                       choices=['baseline', 'proposed', 'both'],
                       help='Which model to train')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Configuration
    config = Config()
    
    # Override config if specified
    if args.epochs is not None:
        config.NUM_EPOCHS = args.epochs
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.lr is not None:
        config.LEARNING_RATE = args.lr
    
    # Set seed
    set_seed(args.seed)
    
    # Create directories
    create_directories([config.DATA_DIR, config.CHECKPOINT_DIR, config.RESULTS_DIR])
    
    # Print configuration
    print("\n" + "="*80)
    print("MEDICAL VQA EXPERIMENT")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Device: {config.DEVICE}")
    print(f"  Epochs: {config.NUM_EPOCHS}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Learning rate: {config.LEARNING_RATE}")
    print(f"  Image size: {config.IMAGE_SIZE}")
    print(f"  Random seed: {args.seed}")
    
    # Create dataloaders
    print("\n" + "-"*80)
    print("Loading datasets...")
    train_loader, val_loader, test_loader, vocab = create_dataloaders(
        config, use_synthetic=True
    )
    
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    if vocab:
        print(f"  Vocabulary size: {len(vocab)}")
    
    # Run experiments
    baseline_results = None
    proposed_results = None
    
    if args.model in ['baseline', 'both']:
        baseline_results = run_baseline_experiment(
            config, train_loader, val_loader, test_loader, vocab
        )
    
    if args.model in ['proposed', 'both']:
        proposed_results = run_proposed_experiment(
            config, train_loader, val_loader, test_loader
        )
    
    # Compare results
    if baseline_results and proposed_results:
        compare_results(baseline_results, proposed_results)
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED")
    print("="*80)
    print(f"\nResults saved to: {config.RESULTS_DIR}")
    print(f"Checkpoints saved to: {config.CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
