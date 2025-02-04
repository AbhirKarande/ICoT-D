import torch
from transformers import Trainer

class DistillationTrainer(Trainer):
    def __init__(self, teacher_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        
    def compute_loss(self, model, inputs, return_outputs=False):
        # Answer prediction loss
        outputs = model(**inputs)
        answer_loss = outputs.loss
        
        # Hidden-state alignment loss
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            
        hidden_loss = torch.mean(
            (outputs.hidden_states[-1] - teacher_outputs.hidden_states[-1])**2
        )
        
        # Combine losses
        total_loss = answer_loss + 0.5 * hidden_loss
        
        return (total_loss, outputs) if return_outputs else total_loss