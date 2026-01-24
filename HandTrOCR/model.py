import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

class SimpleEsposallesTrOCR(nn.Module):
    def __init__(self, max_length=36):
        super(SimpleEsposallesTrOCR, self).__init__()
        self.max_length = max_length
        self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
        self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
        
        # Fix the missing decoder_start_token_id configuration
        if self.model.config.decoder_start_token_id is None:
            self.model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id or self.processor.tokenizer.bos_token_id
        
        # Ensure pad_token_id is set
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        
        # Set eos_token_id if not already set
        if self.model.config.eos_token_id is None:
            self.model.config.eos_token_id = self.processor.tokenizer.eos_token_id

    def forward(self, images, labels):
        """
        Forward pass for training
        
        Args:
            images: Tensor of shape (batch_size, 3, height, width)
            labels: List of ground truth text strings
            
        Returns:
            loss: CrossEntropy loss
            logits: Model output logits
        """
        # Process input labels
        gts = self.processor.tokenizer(
            labels, 
            padding=True, 
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        ).input_ids
        gts = gts.to(images.device)

        # Get model outputs
        outputs = self.model(pixel_values=images, labels=gts)
        logits = outputs.logits

        # Calculate loss manually (ignoring padding tokens)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = gts[..., 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.processor.tokenizer.pad_token_id)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss, logits

    def predict(self, images, early_stopping=True):
        """
        Generate predictions using beam search
        
        Args:
            images: Tensor of shape (batch_size, 3, height, width)
            early_stopping: Whether to stop generation early when EOS is found in all beams
            
        Returns:
            List of predicted text strings
        """
        outputs = self.model.generate(
            pixel_values=images,
            max_length=self.max_length,
            num_beams=4,  # Use beam search for higher quality output
            repetition_penalty=1.2,  # Apply a penalty for repeated tokens
            eos_token_id=self.model.config.eos_token_id,  # Ensure generation stops at the EOS token
            early_stopping=early_stopping  # Stop early when EOS found in all beams
        )
        res = self.processor.batch_decode(outputs, skip_special_tokens=True)
        return res

    def token2text(self, tokens):
        """Convert token IDs to text"""
        token_list = tokens.tolist()
        texts = []
        for token_chars in token_list:
            text = self.processor.tokenizer.decode(token_chars, skip_special_tokens=True)
            texts.append(text)
        return texts

    def decode_logits(self, logits):
        """Decode logits to text using argmax"""
        pred_tokens = torch.argmax(logits, dim=-1)
        pred_texts = self.processor.tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)
        return pred_texts

    def test_model(self):
        """Test model with dummy data"""
        print("=== SIMPLE ESPOSALLES MODEL TEST ===")
        
        # Create dummy images and labels (typical for handwritten words)
        test_image = Image.new('RGB', (224, 224), color='white')
        pixel_values = self.processor(images=[test_image, test_image], return_tensors="pt").pixel_values
        
        # Test labels (typical Esposalles words)
        labels = ["Maria", "Barcelona"]
        
        print(f"Labels: {labels}")
        print(f"Image shape: {pixel_values.shape}")
        
        self.eval()
        with torch.no_grad():
            # Test forward pass
            loss, logits = self.forward(pixel_values, labels)
            print(f"Loss: {loss.item():.6f}")
            print(f"Logits shape: {logits.shape}")
            
            # Test prediction
            predictions = self.predict(pixel_values)
            print(f"Predictions: {predictions}")
        
        print("=== TEST COMPLETE ===\n")

if __name__ == '__main__':
    model = SimpleEsposallesTrOCR()
    model.test_model()