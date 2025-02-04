import torch
from transformers import Trainer, BitsAndBytesConfig

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

class QuantizedDistillationTrainer(DistillationTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _load_model(self, model, args):
        # Load base model with 4-bit quantization
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model.config._name_or_path,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4"
            ),
            device_map="auto"
        )
        return model