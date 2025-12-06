"""Proposed ViT-ClinicalBERT Model for Medical VQA"""

import torch
import torch.nn as nn
from transformers import ViTModel, AutoModel, AutoTokenizer

class ViTClinicalBERTVQA(nn.Module):
    """
    Proposed model using Vision Transformer for image encoding
    and ClinicalBERT for question encoding with cross-attention fusion
    """
    
    def __init__(self, vit_model_name: str = "google/vit-base-patch16-224",
                 text_model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
                 fusion_dim: int = 512, num_attention_heads: int = 8,
                 num_decoder_layers: int = 2, num_answers: int = 1000,
                 dropout: float = 0.1, max_answer_length: int = 30):
        super(ViTClinicalBERTVQA, self).__init__()
        
        self.fusion_dim = fusion_dim
        self.max_answer_length = max_answer_length
        
        # Vision Transformer for image encoding
        self.vit = ViTModel.from_pretrained(vit_model_name)
        self.vit_hidden_dim = self.vit.config.hidden_size  # 768 for ViT-Base
        
        # ClinicalBERT for question encoding
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.text_hidden_dim = self.text_encoder.config.hidden_size  # 768
        
        # Tokenizer for text processing
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        
        # Project to common fusion dimension
        self.image_projection = nn.Linear(self.vit_hidden_dim, fusion_dim)
        self.text_projection = nn.Linear(self.text_hidden_dim, fusion_dim)
        
        # Bidirectional cross-attention for multimodal fusion
        self.cross_attention_v2t = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.cross_attention_t2v = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(fusion_dim)
        self.norm2 = nn.LayerNorm(fusion_dim)
        
        # Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Classification head for closed-ended questions
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, num_answers)
        )
        
        # Decoder for open-ended questions
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=fusion_dim,
            nhead=num_attention_heads,
            dim_feedforward=fusion_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.output_projection = nn.Linear(fusion_dim, num_answers)
        
    def encode_image(self, images):
        """Encode images using Vision Transformer"""
        # ViT forward pass
        outputs = self.vit(pixel_values=images)
        
        # Get patch embeddings (excluding CLS token) and CLS token
        image_features = outputs.last_hidden_state  # (B, num_patches+1, hidden_dim)
        
        # Project to fusion dimension
        image_features = self.image_projection(image_features)  # (B, num_patches+1, fusion_dim)
        
        return image_features
    
    def encode_text(self, questions, attention_mask=None):
        """Encode questions using ClinicalBERT"""
        # ClinicalBERT forward pass
        outputs = self.text_encoder(
            input_ids=questions,
            attention_mask=attention_mask
        )
        
        # Get token embeddings
        text_features = outputs.last_hidden_state  # (B, seq_len, hidden_dim)
        
        # Project to fusion dimension
        text_features = self.text_projection(text_features)  # (B, seq_len, fusion_dim)
        
        return text_features
    
    def forward(self, images, questions, attention_mask=None, mode='classify'):
        """
        Args:
            images: (batch_size, 3, 224, 224)
            questions: (batch_size, max_length) - token IDs from tokenizer
            attention_mask: (batch_size, max_length)
            mode: 'classify' for closed-ended, 'generate' for open-ended
        
        Returns:
            logits or generated sequences
        """
        # Encode image and text
        image_features = self.encode_image(images)  # (B, num_patches, fusion_dim)
        text_features = self.encode_text(questions, attention_mask)  # (B, seq_len, fusion_dim)
        
        # Bidirectional cross-attention
        # Vision to Text: use text as query, image as key/value
        v2t_attended, _ = self.cross_attention_v2t(
            text_features,   # query
            image_features,  # key
            image_features   # value
        )  # (B, seq_len, fusion_dim)
        v2t_attended = self.norm1(text_features + v2t_attended)
        
        # Text to Vision: use image as query, text as key/value
        t2v_attended, _ = self.cross_attention_t2v(
            image_features,  # query
            text_features,   # key
            text_features    # value
        )  # (B, num_patches, fusion_dim)
        t2v_attended = self.norm2(image_features + t2v_attended)
        
        # Aggregate features (mean pooling)
        v2t_pooled = v2t_attended.mean(dim=1)  # (B, fusion_dim)
        t2v_pooled = t2v_attended.mean(dim=1)  # (B, fusion_dim)
        
        # Fuse both directions
        fused = torch.cat([v2t_pooled, t2v_pooled], dim=1)  # (B, fusion_dim*2)
        fused_features = self.fusion_mlp(fused)  # (B, fusion_dim)
        
        if mode == 'classify':
            # Classification for closed-ended questions
            logits = self.classifier(fused_features)  # (B, num_answers)
            return logits
        else:
            # For generation, return logits (simplified)
            return self.classifier(fused_features)
    
    def generate_answer(self, images, questions, attention_mask=None, max_length=30):
        """Generate answer for open-ended questions"""
        with torch.no_grad():
            # Get fused features
            image_features = self.encode_image(images)
            text_features = self.encode_text(questions, attention_mask)
            
            # Bidirectional cross-attention and fusion (same as forward)
            v2t_attended, _ = self.cross_attention_v2t(
                text_features, image_features, image_features
            )
            v2t_attended = self.norm1(text_features + v2t_attended)
            
            t2v_attended, _ = self.cross_attention_t2v(
                image_features, text_features, text_features
            )
            t2v_attended = self.norm2(image_features + t2v_attended)
            
            v2t_pooled = v2t_attended.mean(dim=1)
            t2v_pooled = t2v_attended.mean(dim=1)
            
            fused = torch.cat([v2t_pooled, t2v_pooled], dim=1)
            fused_features = self.fusion_mlp(fused)
            
            # Simple prediction (argmax)
            logits = self.classifier(fused_features)
            predictions = torch.argmax(logits, dim=1)
            
            return predictions
    
    def get_attention_maps(self, images, questions, attention_mask=None):
        """Get attention maps for visualization"""
        with torch.no_grad():
            image_features = self.encode_image(images)
            text_features = self.encode_text(questions, attention_mask)
            
            # Get attention weights from cross-attention
            _, v2t_weights = self.cross_attention_v2t(
                text_features, image_features, image_features
            )
            
            _, t2v_weights = self.cross_attention_t2v(
                image_features, text_features, text_features
            )
            
            return {
                'v2t_attention': v2t_weights,  # (B, seq_len, num_patches)
                't2v_attention': t2v_weights   # (B, num_patches, seq_len)
            }


def count_parameters(model):
    """Count trainable parameters"""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    vit = sum(p.numel() for p in model.vit.parameters() if p.requires_grad)
    text = sum(p.numel() for p in model.text_encoder.parameters() if p.requires_grad)
    fusion = total - vit - text
    
    return {
        'total': total,
        'vit': vit,
        'text_encoder': text,
        'fusion_decoder': fusion
    }


if __name__ == "__main__":
    # Test the model
    print("Initializing ViT-ClinicalBERT model...")
    model = ViTClinicalBERTVQA(num_answers=100)
    
    # Dummy inputs
    images = torch.randn(2, 3, 224, 224)
    questions = torch.randint(0, 30522, (2, 50))  # BERT vocab size
    attention_mask = torch.ones(2, 50)
    
    # Forward pass
    print("Running forward pass...")
    outputs = model(images, questions, attention_mask, mode='classify')
    print(f"Output shape: {outputs.shape}")
    
    # Count parameters
    params = count_parameters(model)
    print(f"\nParameter counts:")
    print(f"  Total: {params['total']:,}")
    print(f"  ViT: {params['vit']:,}")
    print(f"  Text Encoder: {params['text_encoder']:,}")
    print(f"  Fusion+Decoder: {params['fusion_decoder']:,}")
