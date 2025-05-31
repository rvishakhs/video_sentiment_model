from models import MultimodalSentimentalModel


def count_params(model):
    """
    Count the number of trainable parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to count parameters for.

    Returns:
        int: The total number of trainable parameters in the model.
    """
    params_dict = {
        'text_encoder' : 0,
        'video_encoder' : 0,
        'audio_encoder' : 0,
        'fusion_module' : 0,
        'emotional_classifier' : 0,
        'sentimental_classifier' : 0
    }

    total_params = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            total_params += param_count

            if 'text_encoder' in name:
                params_dict['text_encoder'] += param_count
            elif 'video_encoder' in name:
                params_dict['video_encoder'] += param_count
            elif 'audio_encoder' in name:
                params_dict['audio_encoder'] += param_count
            elif 'fusion_layer' in name: 
                params_dict['fusion_module'] += param_count
            elif 'emotion_classifier' in name:
                params_dict['emotional_classifier'] += param_count
            elif 'sentiment_classifier' in name:
                params_dict['sentimental_classifier'] += param_count

    return params_dict, total_params

if __name__ == "__main__":
    model = MultimodalSentimentalModel()
    param_dict, total_params = count_params(model)

    print(f"Parameter count by component")
    for component, count in param_dict.items():
        print(f"{component:20s}: {count:,} parameters")

    print(f"\nTotal number of trainable parameters: {total_params:,}")


