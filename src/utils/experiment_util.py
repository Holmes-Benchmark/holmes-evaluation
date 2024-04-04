import torch
import transformers
from tokenizers import Tokenizer
from transformers import SLOW_TO_FAST_CONVERTERS, PreTrainedTokenizer

def convert_slow_tokenizer(transformer_tokenizer) -> Tokenizer:
    """
    Utilities to convert a slow tokenizer instance in a fast tokenizer instance.

    Args:
        transformer_tokenizer ([`~tokenization_utils_base.PreTrainedTokenizer`]):
            Instance of a slow tokenizer to convert in the backend tokenizer for
            [`~tokenization_utils_base.PreTrainedTokenizerFast`].

    Return:
        A instance of [`~tokenizers.Tokenizer`] to be used as the backend tokenizer of a
        [`~tokenization_utils_base.PreTrainedTokenizerFast`]
    """

    tokenizer_class_name = transformer_tokenizer.__class__.__name__

    if tokenizer_class_name not in SLOW_TO_FAST_CONVERTERS:
        raise ValueError(
            f"An instance of tokenizer class {tokenizer_class_name} cannot be converted in a Fast tokenizer instance."
            " No converter was found. Currently available slow->fast convertors:"
            f" {list(SLOW_TO_FAST_CONVERTERS.keys())}"
        )

    converter_class = SLOW_TO_FAST_CONVERTERS[tokenizer_class_name]

    return converter_class(transformer_tokenizer).converted()


def update_tokenizer(tokenizer):
    if PreTrainedTokenizer in type(tokenizer).__bases__:
        return convert_slow_tokenizer(tokenizer)
    else:
        return tokenizer




def randomize_model(model):
    for module_ in model.named_modules():
        if isinstance(module_[1],(torch.nn.Linear, torch.nn.Conv1d, torch.nn.Embedding)):
            torch.nn.init.xavier_uniform_(module_[1])
        elif isinstance(module_[1], torch.nn.LayerNorm):
            module_[1].bias.data.zero_()
            module_[1].weight.data.fill_(1.0)
        if isinstance(module_[1], torch.nn.Linear) and module_[1].bias is not None:
            module_[1].bias.data.zero_()
    return model

def init_random_weights(module: torch.nn.Module) -> None:
    """Based on Jawahar et al., 2019

    https://github.com/ganeshjawahar/interpret_bert/blob/master/probing/extract_features.py"""
    if type(module) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.01)
        #self._total += module.weight.size(0) * module.weight.size(1)
        #self._total += module.bias.size(0)
    elif type(module) == torch.nn.Embedding:
        torch.nn.init.xavier_uniform_(module.weight)
        #self._total += module.weight.size(0) * module.weight.size(1)
    elif type(module) == torch.nn.LayerNorm:
        module.weight.data.fill_(1.0)
    elif type(module) == transformers.modeling_utils.Conv1D:
        torch.nn.init.xavier_uniform_(module.weight)
    elif hasattr(module, 'weight') and hasattr(module, 'bias'):
        print()
        #self._total += module.weight.size(0)
        #self._total += module.bias.size(0)
