import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoModelForSequenceClassification
from typing import List, Optional


class MoEClassifier(nn.Module):
    """Mixture-of-Experts for regression, with each expert being a pretrained transformer model."""
    def __init__(
        self,
        experts: List[str],
        encoder: Optional[str] = None,
    ):
        '''
        Args:
            experts: List of hf model names to be used as experts (each with its own head).
            encoder: Optional encoder name for the gating mechanism (default = first expert).
        '''
        super(MoEClassifier, self).__init__() 
    
        self.experts = nn.ModuleList([
            AutoModelForSequenceClassification.from_pretrained(expert, num_labels=3)
            for expert in experts
        ])
        encoder = encoder or experts[0]
        self.encoder = AutoModel.from_pretrained(encoder)
        
        encoder_dim = self.encoder.config.hidden_size
        self.gate = nn.Linear(encoder_dim, len(experts))
        
        
    def forward(self, input_ids, attention_mask):        
        with torch.no_grad():
            hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls_embed = hidden.last_hidden_state[:, 0]
        
        gate_scores = F.gumbel_softmax(self.gate(cls_embed), dim=-1, hard=True)
            
        expert_out = torch.stack([
            expert(input_ids, attention_mask).logits
            for expert in self.experts
        ], dim=1)
        
        output = (gate_scores.unsqueeze(-1) * expert_out).sum(dim=1)
        return output
