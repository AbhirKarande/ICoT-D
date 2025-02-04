from peft import LoraConfig, get_peft_model

def create_lora_adapter(base_model, adapter_type: str):
    config = LoraConfig(
        r=64,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        adapter_name=adapter_type
    )
    return get_peft_model(base_model, config)