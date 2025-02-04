from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig
import torch

def create_qlora_adapter(base_model, adapter_type: str):
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    
    # QLoRA config
    config = LoraConfig(
        r=64,  # Reduced from original 64 to 32 for quantization
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        adapter_name=adapter_type,
        quantization_config=bnb_config  # Add quantization config
    )
    
    return get_peft_model(base_model, config)