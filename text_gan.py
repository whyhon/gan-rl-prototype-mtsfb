"""
Simplified Text GAN for Adversarial Prompt Generation
Lightweight implementation for generating text-based adversarial inputs
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from typing import List, Dict, Tuple, Optional
import logging
from config_loader import ConfigurationLoader

class SimpleTextGenerator(nn.Module):
    """Simplified generator for creating adversarial text prompts"""
    
    def __init__(self, vocab_size: int = 1000, embedding_dim: int = 64, 
                 hidden_dim: int = 128, max_length: int = 50):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)
        lstm_out = self.dropout(lstm_out)
        output = self.output_layer(lstm_out)
        return output, hidden

class SimpleTextDiscriminator(nn.Module):
    """Simplified discriminator for evaluating text quality"""
    
    def __init__(self, vocab_size: int = 1000, embedding_dim: int = 64, 
                 hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embedded)
        # Use last hidden state
        output = self.classifier(hidden[-1])
        return output.squeeze()

class AdversarialTextGAN:
    """Main GAN class for generating adversarial text inputs"""
    
    def __init__(self, vocab_size: int = None, device: str = None, config_dir: str = "config"):
        self.config_loader = ConfigurationLoader(config_dir)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)

        # Initialize vocabulary first to determine actual vocab size
        self.init_vocabulary()

        # Use calculated vocab size if not provided
        if vocab_size is None:
            vocab_size = len(self.vocab)
        self.vocab_size = vocab_size

        # Initialize networks
        self.generator = SimpleTextGenerator(vocab_size).to(self.device)
        self.discriminator = SimpleTextDiscriminator(vocab_size).to(self.device)

        # Optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002)
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002)

        # Loss function
        self.criterion = nn.BCELoss()
        
    def init_vocabulary(self):
        """Initialize multilingual vocabulary from external configuration"""
        try:
            # Load combined vocabulary from all languages
            combined_vocab = self.config_loader.get_combined_vocabulary(include_adversarial=True)

            # Add numbers and special tokens
            numbers = [str(i) for i in range(100)]
            special = ["<start>", "<end>", "<unk>", "<pad>"]

            # Combine all vocabulary sources
            self.vocab = combined_vocab + numbers + special

            # Remove duplicates while preserving order
            self.vocab = list(dict.fromkeys(self.vocab))

            # Ensure minimum vocabulary size
            min_vocab_size = 1000
            while len(self.vocab) < min_vocab_size:
                self.vocab.append(f"word_{len(self.vocab)}")

            self.logger.info(f"Initialized vocabulary with {len(self.vocab)} words")

        except Exception as e:
            self.logger.error(f"Failed to load external vocabulary, using fallback: {e}")
            # Fallback to basic vocabulary
            self._init_fallback_vocabulary()

        # Create word-to-index mappings
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}

    def _init_fallback_vocabulary(self):
        """Fallback vocabulary initialization if config loading fails"""
        base_words = [
            "what", "how", "why", "when", "where", "who", "tell", "me", "about",
            "explain", "describe", "show", "give", "provide", "create", "make",
            "write", "generate", "help", "assist", "please", "can", "you", "I",
            "the", "a", "an", "is", "are", "was", "were", "have", "has", "do",
            "does", "will", "would", "could", "should", "must", "may", "might"
        ]

        adversarial_words = [
            "ignore", "forget", "bypass", "override", "hack", "break", "exploit",
            "jailbreak", "pretend", "roleplay", "assume", "imagine", "hypothetical",
            "secret", "confidential", "hidden", "classified", "sensitive"
        ]

        numbers = [str(i) for i in range(100)]
        special = ["<start>", "<end>", "<unk>", "<pad>"]

        self.vocab = base_words + adversarial_words + numbers + special
        
    def text_to_tensor(self, text: str, max_length: int = 50) -> torch.Tensor:
        """Convert text to tensor indices"""
        words = text.lower().split()
        indices = []
        
        for word in words[:max_length-1]:
            indices.append(self.word_to_idx.get(word, self.word_to_idx["<unk>"]))
        
        # Pad sequence
        while len(indices) < max_length:
            indices.append(self.word_to_idx["<pad>"])
            
        return torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(self.device)
    
    def tensor_to_text(self, tensor: torch.Tensor) -> str:
        """Convert tensor indices back to text"""
        indices = tensor.cpu().numpy().flatten()
        words = []
        
        for idx in indices:
            word = self.idx_to_word.get(idx, "<unk>")
            if word not in ["<pad>", "<start>", "<end>"]:
                words.append(word)
            elif word == "<end>":
                break
                
        return " ".join(words)
    
    def generate_adversarial_prompt(self, seed_text: str = None, 
                                  temperature: float = 1.0) -> str:
        """Generate an adversarial text prompt"""
        self.generator.eval()
        
        try:
            with torch.no_grad():
                if seed_text:
                    input_tensor = self.text_to_tensor(seed_text, max_length=10)
                else:
                    # Start with random seed
                    start_idx = self.word_to_idx.get("<start>", 0)
                    input_tensor = torch.tensor([[start_idx]], dtype=torch.long).to(self.device)
                
                hidden = None
                generated = []
                
                for _ in range(30):  # Generate up to 30 tokens
                    try:
                        output, hidden = self.generator(input_tensor, hidden)
                        
                        # Apply temperature sampling
                        logits = output[:, -1, :] / temperature
                        probabilities = torch.softmax(logits, dim=-1)
                        
                        # Sample from distribution
                        next_token = torch.multinomial(probabilities, 1)
                        generated.append(next_token.item())
                        
                        # Use generated token as next input
                        input_tensor = next_token
                        
                        # Stop if we generate end token
                        if next_token.item() == self.word_to_idx.get("<end>", -1):
                            break
                    except Exception as e:
                        self.logger.warning(f"Token generation failed: {e}")
                        break
                
                # Convert to text
                if generated:
                    generated_tensor = torch.tensor(generated).unsqueeze(0)
                    return self.tensor_to_text(generated_tensor)
                else:
                    return "what is"  # Fallback prompt
        except Exception as e:
            self.logger.error(f"Prompt generation failed: {e}")
            return "what is"  # Fallback prompt
    
    def generate_batch_prompts(self, batch_size: int = 5, 
                             temperature: float = 1.0) -> List[str]:
        """Generate a batch of adversarial prompts"""
        prompts = []
        for _ in range(batch_size):
            prompt = self.generate_adversarial_prompt(temperature=temperature)
            if prompt.strip():  # Only add non-empty prompts
                prompts.append(prompt.strip())
        return prompts
    
    def simple_train_step(self, real_texts: List[str], epochs: int = 1):
        """Simplified training step (for demonstration)"""
        for epoch in range(epochs):
            # Convert real texts to tensors
            real_tensors = [self.text_to_tensor(text) for text in real_texts]
            
            # Train discriminator
            self.d_optimizer.zero_grad()
            
            # Real data
            try:
                real_batch = torch.cat(real_tensors, dim=0)
                real_labels = torch.ones(real_batch.size(0)).to(self.device)
                real_pred = self.discriminator(real_batch)
                real_loss = self.criterion(real_pred, real_labels)
                
                # Fake data
                fake_prompts = self.generate_batch_prompts(len(real_texts))
                fake_tensors = [self.text_to_tensor(prompt) for prompt in fake_prompts if prompt.strip()]
                if not fake_tensors:  # Skip if no valid prompts generated
                    continue
                fake_batch = torch.cat(fake_tensors, dim=0)
                fake_labels = torch.zeros(fake_batch.size(0)).to(self.device)
                fake_pred = self.discriminator(fake_batch)
                fake_loss = self.criterion(fake_pred, fake_labels)
                
                d_loss = real_loss + fake_loss
                d_loss.backward()
                self.d_optimizer.step()
                
                # Train generator
                self.g_optimizer.zero_grad()
                fake_prompts = self.generate_batch_prompts(len(real_texts))
                fake_tensors = [self.text_to_tensor(prompt) for prompt in fake_prompts if prompt.strip()]
                if not fake_tensors:  # Skip if no valid prompts generated
                    continue
                fake_batch = torch.cat(fake_tensors, dim=0)
                fake_pred = self.discriminator(fake_batch)
                g_loss = self.criterion(fake_pred, torch.ones_like(fake_pred))
                g_loss.backward()
                self.g_optimizer.step()
            except Exception as e:
                self.logger.warning(f"Training step failed: {e}, skipping epoch {epoch}")
            
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}: D_loss={d_loss:.4f}, G_loss={g_loss:.4f}")

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize GAN
    gan = AdversarialTextGAN(vocab_size=500)
    
    # Generate some sample adversarial prompts
    print("Generated adversarial prompts:")
    for i in range(5):
        prompt = gan.generate_adversarial_prompt(temperature=0.8)
        print(f"{i+1}: {prompt}")
    
    # Example training with simple texts
    sample_texts = [
        "what is artificial intelligence",
        "explain machine learning concepts", 
        "how does neural network work",
        "describe deep learning models"
    ]
    
    print("\nTraining GAN with sample texts...")
    gan.simple_train_step(sample_texts, epochs=50)