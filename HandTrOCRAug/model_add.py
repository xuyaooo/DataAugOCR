import os
import torch
import torch.nn as nn
from transformers import (
    TrOCRProcessor, 
    VisionEncoderDecoderModel,
    BertModel, 
    BertTokenizer,
)
from transformers.modeling_outputs import BaseModelOutput

# CUDA device should be set via environment variable or command line
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # Commented out - set via environment


def set_dropout_rate(model, new_rate):
    """Set dropout rate for all dropout layers in the model."""
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            print(f"Setting dropout for {name}: {module.p} -> {new_rate}")
            module.p = new_rate
            count += 1
    print(f"Total dropout layers modified: {count}")


# =================================================================================
#  MODEL DEFINITION (Simple dropout setting)
# =================================================================================
class ModifiedViTEncoder(nn.Module):
    """
    Modified ViT encoder with cross-attention layers and dimension reduction
    """
    def __init__(self, vit_model, hidden_size=768, cross_attn_dim=384, num_heads=6, dropout_rate=0.3):
        super(ModifiedViTEncoder, self).__init__()
        
        self.embeddings = vit_model.embeddings
        self.original_layers = vit_model.encoder.layer
        self.layernorm = vit_model.layernorm
        self.pooler = vit_model.pooler if hasattr(vit_model, 'pooler') else None
        
        # CHANGE 1: Reduce to 1 layer only (based on your results)
        self.cross_attention_positions = [9]  # Keep only the deepest layer
        
        # CHANGE 2: Add projection layers for dimension reduction
        self.cross_attn_proj_down = nn.Linear(hidden_size, cross_attn_dim)
        self.text_proj_down = nn.Linear(hidden_size, cross_attn_dim)
        self.cross_attn_proj_up = nn.Linear(cross_attn_dim, hidden_size)
        
        # CHANGE 3: Smaller cross-attention with higher dropout
        self.cross_attentions = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=cross_attn_dim,    # REDUCED: 384 instead of 768
                num_heads=num_heads,         # REDUCED: 6 instead of 12
                dropout=dropout_rate,        # INCREASED: 0.3 instead of 0.1
                batch_first=True
            ) for _ in self.cross_attention_positions
        ])
        
        
    def forward(self, pixel_values, text_features, text_attention_mask=None):
        embedding_output = self.embeddings(pixel_values)
        hidden_states = embedding_output
        cross_attn_idx = 0
        
        for layer_idx, layer in enumerate(self.original_layers):
            layer_outputs = layer(hidden_states)
            hidden_states = layer_outputs[0]
            
            if (layer_idx + 1) in self.cross_attention_positions:
                # CHANGE 5: Project to smaller dimension before cross-attention
                hidden_proj = self.cross_attn_proj_down(hidden_states)      # 768 → 384
                text_proj = self.text_proj_down(text_features)        # 768 → 384
                
                # Cross-attention in reduced dimension space
                key_padding_mask = (text_attention_mask == 0) if text_attention_mask is not None else None
                cross_attn_output, _ = self.cross_attentions[cross_attn_idx](
                    query=hidden_proj, 
                    key=text_proj, 
                    value=text_proj, 
                    key_padding_mask=key_padding_mask
                )
                
                
                # CHANGE 7: Project back to original dimension
                cross_attn_output = self.cross_attn_proj_up(cross_attn_output)  # 384 → 768
                
                # Residual connection and normalization
                hidden_states = hidden_states + cross_attn_output
                cross_attn_idx += 1
        
        hidden_states = self.layernorm(hidden_states)
        return hidden_states


class ImageTextTrOCRModified(nn.Module):
    def __init__(self, max_length=36, bert_model_name='bert-base-uncased', dropout_rate=0.1):
        super(ImageTextTrOCRModified, self).__init__()
        self.max_length = max_length
        
        self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
        self.original_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
        
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert_model = BertModel.from_pretrained(bert_model_name)
        
        original_vit = self.original_model.get_encoder()
        
        # CHANGE 8: Pass the new parameters to ModifiedViTEncoder
        self.modified_encoder = ModifiedViTEncoder(
            original_vit, 
            hidden_size=768,
            cross_attn_dim=384,      # NEW: Reduced dimension
            num_heads=4,             # NEW: Fewer heads
            dropout_rate=dropout_rate
        )
        
        self.decoder = self.original_model.get_decoder()
        self._fix_token_configs()
        
        # CHANGE 9: Increased dropout rate
        set_dropout_rate(self, dropout_rate)  # Now 0.3 instead of 0.1
        
        # CHANGE 10: Freeze BERT to prevent overfitting
        for param in self.bert_model.parameters():
            param.requires_grad = False
        self.bert_model.eval()

        
    def _fix_token_configs(self):
        if self.decoder.config.decoder_start_token_id is None:
            self.decoder.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id
        if self.decoder.config.pad_token_id is None:
            self.decoder.config.pad_token_id = self.processor.tokenizer.pad_token_id
        if self.decoder.config.eos_token_id is None:
            self.decoder.config.eos_token_id = self.processor.tokenizer.sep_token_id
        self.original_model.config.decoder_start_token_id = self.decoder.config.decoder_start_token_id
        self.original_model.config.pad_token_id = self.decoder.config.pad_token_id
        self.original_model.config.eos_token_id = self.decoder.config.eos_token_id

    def encode_text(self, text_inputs):
        device = next(self.parameters()).device
        text_tokens = self.bert_tokenizer(
            text_inputs, padding=True, truncation=True, max_length=128, return_tensors='pt'
        ).to(device)
        with torch.no_grad():
            bert_outputs = self.bert_model(**text_tokens)
            text_features = bert_outputs.last_hidden_state
            text_attention_mask = text_tokens['attention_mask']
        return text_features, text_attention_mask
    
    def forward(self, pixel_values, labels, text_inputs=None):
        """
        Forward pass that expects already processed pixel_values
        
        Args:
            pixel_values: Already processed image tensors (batch_size, 3, 384, 384)
            labels: Ground truth text labels
            text_inputs: Text questions/prompts
        """
        device = next(self.parameters()).device
        pixel_values = pixel_values.to(device)
        
        if text_inputs is None:
            text_inputs = ["What is this word?"] * len(labels)
        
        text_features, text_attention_mask = self.encode_text(text_inputs)
        
        gts = self.processor.tokenizer(
            labels, padding="longest", truncation=True, max_length=self.max_length, return_tensors='pt'
        ).input_ids.to(device)
        
        encoder_outputs = self.modified_encoder(
            pixel_values=pixel_values, text_features=text_features, text_attention_mask=text_attention_mask
        )
        
        batch_size, _ = gts.shape
        decoder_start_token_id = self.decoder.config.decoder_start_token_id
        start_tokens = torch.full((batch_size, 1), decoder_start_token_id, dtype=torch.long, device=device)
        gts_body = gts[:, :-1]
        decoder_input_ids = torch.cat([start_tokens, gts_body], dim=1)
        decoder_attention_mask = (decoder_input_ids != self.decoder.config.pad_token_id).long()

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids, attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs, encoder_attention_mask=None,
            labels=gts, output_hidden_states=False, return_dict=True
        )
        
        return decoder_outputs.loss, decoder_outputs.logits

    def predict(self, pixel_values, text_inputs=None):
        """
        Prediction method that expects already processed pixel_values
        
        Args:
            pixel_values: Already processed image tensors (batch_size, 3, 384, 384)  
            text_inputs: Text questions/prompts
        """
        device = next(self.parameters()).device
        pixel_values = pixel_values.to(device)
        
        # Handle case where pixel_values might be a list of PIL images (fallback)
        if isinstance(pixel_values, list):
            pixel_values = self.processor(images=pixel_values, return_tensors="pt").pixel_values.to(device)
        
        if text_inputs is None:
            batch_size = pixel_values.shape[0]
            text_inputs = ["What is this word?"] * batch_size
        
        text_features, text_attention_mask = self.encode_text(text_inputs)
        
        encoder_hidden_states = self.modified_encoder(
            pixel_values=pixel_values, text_features=text_features, text_attention_mask=text_attention_mask
        )
        
        encoder_outputs_structured = BaseModelOutput(last_hidden_state=encoder_hidden_states)
        
        outputs = self.original_model.generate(
            encoder_outputs=encoder_outputs_structured, max_length=self.max_length,
            num_beams=4, early_stopping=True, repetition_penalty=1.2
        )
        
        return self.processor.batch_decode(outputs, skip_special_tokens=True)