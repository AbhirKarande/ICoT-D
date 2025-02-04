def fuse_adapters(adapters, weights):
    fused_weights = {}
    for adapter in adapters:
        for name, param in adapter.named_parameters():
            if name not in fused_weights:
                fused_weights[name] = weights[adapter.config.adapter_name] * param
            else:
                fused_weights[name] += weights[adapter.config.adapter_name] * param
    return fused_weights