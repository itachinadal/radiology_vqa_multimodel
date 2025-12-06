"""Baseline CNN-LSTM Model for Medical VQA"""

import torch
import torch.nn as nn
import torchvision.models as models

class BaselineCNNLSTM(nn.Module):
    """
    Baseline model using CNN (ResNet50) for image encoding 
    and LSTM for question encoding
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 300,
                 lstm_hidden_dim: int = 512, lstm_num_layers: int = 2,
                 cnn_feature_dim: int = 2048, fusion_dim: int = 512,
                 num_answers: int = 1000, dropout: float = 0.1):
        super(BaselineCNNLSTM, self).__init__()
        
        self.vocab_size = vocab_size
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        
        # Image encoder: ResNet50 (pretrained on ImageNet)
        resnet = models.resnet50(pretrained=True)
        # Remove the final classification layer
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.image_projection = nn.Linear(cnn_feature_dim, fusion_dim)
        
        # Question encoder: Embedding + LSTM
        self.question_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.question_lstm = nn.LSTM(
            embedding_dim,
            lstm_hidden_dim,
            lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=True
        )
        self.question_projection = nn.Linear(lstm_hidden_dim * 2, fusion_dim)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Fusion and classification
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classification head for closed-ended questions
        self.classifier = nn.Linear(fusion_dim, num_answers)
        
        # Decoder for open-ended questions (simplified)
        self.decoder_rnn = nn.LSTM(
            embedding_dim + fusion_dim,
            lstm_hidden_dim,
            1,
            batch_first=True
        )
        self.decoder_output = nn.Linear(lstm_hidden_dim, vocab_size)
    
    def forward(self, images, questions, mode='classify'):
        """
        Args:
            images: (batch_size, 3, 224, 224)
            questions: (batch_size, max_length)
            mode: 'classify' for closed-ended, 'generate' for open-ended
        
        Returns:
            logits or generated sequences
        """
        batch_size = images.size(0)
        
        # Encode image
        image_features = self.image_encoder(images)  # (B, 2048, 1, 1)
        image_features = image_features.squeeze(-1).squeeze(-1)  # (B, 2048)
        image_features = self.image_projection(image_features)  # (B, fusion_dim)
        image_features = image_features.unsqueeze(1)  # (B, 1, fusion_dim)
        
        # Encode question
        question_emb = self.question_embedding(questions)  # (B, L, embedding_dim)
        lstm_out, (hidden, cell) = self.question_lstm(question_emb)  # (B, L, hidden*2)
        # Take the last hidden state
        question_features = lstm_out[:, -1, :]  # (B, hidden*2)
        question_features = self.question_projection(question_features)  # (B, fusion_dim)
        question_features = question_features.unsqueeze(1)  # (B, 1, fusion_dim)
        
        # Attention-based fusion
        attended_features, _ = self.attention(
            question_features,  # query
            image_features,     # key
            image_features      # value
        )  # (B, 1, fusion_dim)
        
        # Concatenate and fuse
        fused = torch.cat([question_features.squeeze(1), 
                          attended_features.squeeze(1)], dim=1)  # (B, fusion_dim*2)
        fused_features = self.fusion(fused)  # (B, fusion_dim)
        
        if mode == 'classify':
            # Classification for closed-ended questions
            logits = self.classifier(fused_features)  # (B, num_answers)
            return logits
        else:
            # Generation for open-ended questions (simplified)
            # This is a placeholder - full implementation would use teacher forcing
            return self.classifier(fused_features)
    
    def generate_answer(self, images, questions, max_length=30, vocab=None):
        """Generate answer for open-ended questions"""
        batch_size = images.size(0)
        
        # Get fused features (same as forward pass)
        with torch.no_grad():
            # Encode image
            image_features = self.image_encoder(images)
            image_features = image_features.squeeze(-1).squeeze(-1)
            image_features = self.image_projection(image_features)
            image_features = image_features.unsqueeze(1)
            
            # Encode question
            question_emb = self.question_embedding(questions)
            lstm_out, (hidden, cell) = self.question_lstm(question_emb)
            question_features = lstm_out[:, -1, :]
            question_features = self.question_projection(question_features)
            question_features = question_features.unsqueeze(1)
            
            # Attention and fusion
            attended_features, _ = self.attention(
                question_features, image_features, image_features
            )
            fused = torch.cat([question_features.squeeze(1), 
                              attended_features.squeeze(1)], dim=1)
            fused_features = self.fusion(fused)
            
            # Simple argmax prediction (not true generation)
            logits = self.classifier(fused_features)
            predictions = torch.argmax(logits, dim=1)
            
            return predictions


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    vocab_size = 5000
    model = BaselineCNNLSTM(vocab_size=vocab_size, num_answers=100)
    
    # Dummy inputs
    images = torch.randn(2, 3, 224, 224)
    questions = torch.randint(0, vocab_size, (2, 50))
    
    # Forward pass
    outputs = model(images, questions, mode='classify')
    print(f"Output shape: {outputs.shape}")
    print(f"Total parameters: {count_parameters(model):,}")
