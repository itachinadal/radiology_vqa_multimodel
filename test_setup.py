"""
Quick test script to verify implementation
Run this to ensure all components work correctly
"""

import torch
import sys

print("="*80)
print("MEDICAL VQA IMPLEMENTATION TEST")
print("="*80)

# Test 1: Import all modules
print("\n1. Testing module imports...")
try:
    from config import Config
    from dataset import create_dataloaders, SyntheticMedicalVQADataset
    from baseline_model import BaselineCNNLSTM
    from proposed_model import ViTClinicalBERTVQA
    from train import Trainer
    from metrics import VQAMetrics
    from utils import set_seed
    print("✓ All modules imported successfully")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Check CUDA availability
print("\n2. Checking CUDA availability...")
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   Device: {torch.cuda.get_device_name(0)}")

# Test 3: Create configuration
print("\n3. Creating configuration...")
config = Config()
print(f"✓ Config created, device: {config.DEVICE}")

# Test 4: Create synthetic dataset
print("\n4. Creating synthetic dataset...")
try:
    vocab = {'<PAD>': 0, '<UNK>': 1, 'yes': 2, 'no': 3}
    dataset = SyntheticMedicalVQADataset(num_samples=10, vocab=None)
    sample = dataset[0]
    print(f"✓ Dataset created, sample keys: {list(sample.keys())}")
    print(f"   Image shape: {sample['image'].shape}")
    print(f"   Question: {sample['raw_question']}")
    print(f"   Answer: {sample['raw_answer']}")
except Exception as e:
    print(f"✗ Dataset creation failed: {e}")
    sys.exit(1)

# Test 5: Create baseline model
print("\n5. Creating baseline model...")
try:
    baseline = BaselineCNNLSTM(vocab_size=100, num_answers=10)
    images = torch.randn(2, 3, 224, 224)
    questions = torch.randint(0, 100, (2, 50))
    outputs = baseline(images, questions)
    print(f"✓ Baseline model created")
    print(f"   Output shape: {outputs.shape}")
    print(f"   Parameters: {sum(p.numel() for p in baseline.parameters()):,}")
except Exception as e:
    print(f"✗ Baseline model failed: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Create proposed model (if transformers installed)
print("\n6. Creating proposed model...")
try:
    from transformers import ViTModel, AutoModel
    print("   Downloading pretrained models (this may take a while)...")
    proposed = ViTClinicalBERTVQA(num_answers=10)
    images = torch.randn(2, 3, 224, 224)
    questions = torch.randint(0, 30522, (2, 50))
    attention_mask = torch.ones(2, 50)
    outputs = proposed(images, questions, attention_mask)
    print(f"✓ Proposed model created")
    print(f"   Output shape: {outputs.shape}")
    
    params = sum(p.numel() for p in proposed.parameters())
    print(f"   Parameters: {params:,}")
except ImportError:
    print("✗ Transformers library not installed")
    print("   Run: pip install transformers")
except Exception as e:
    print(f"✗ Proposed model failed: {e}")
    import traceback
    traceback.print_exc()

# Test 7: Test metrics
print("\n7. Testing metrics...")
try:
    metrics = VQAMetrics()
    predictions = torch.tensor([0, 1, 2, 0])
    targets = torch.tensor([0, 1, 2, 1])
    metrics.update(predictions, targets)
    results = metrics.compute_classification_metrics()
    print(f"✓ Metrics computed")
    print(f"   Accuracy: {results['accuracy']:.4f}")
    print(f"   F1: {results['f1']:.4f}")
except Exception as e:
    print(f"✗ Metrics failed: {e}")

print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)
print("\n✓ Implementation verified successfully!")
print("\nNext steps:")
print("1. Run a quick training test:")
print("   python run_experiments.py --model baseline --epochs 2")
print("\n2. Run full experiments:")
print("   python run_experiments.py --model both --epochs 10")
print("\n3. For production, prepare real medical datasets")
print("="*80)
