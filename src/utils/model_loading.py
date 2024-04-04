import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer

from defs.control_task_types import CONTROL_TASK_TYPES
from model.EightBitTransformer import EightBitTransformer
from model.FourBitTransformer import FourBitTransformer
from model.HalfPrecisionTransformer import HalfPrecisionTransformer
from model.ParallelSentenceTransformer import ParallelSentenceTransformer
from model.transformer_adaptions import BartTransformer
from utils.SpecificLayerPooling import SpecificLayerPooling
from utils.experiment_util import init_random_weights


def load_model(model_name, control_task_type, encoding, scalar_mixin=False):

    if "glove" in model_name:
        base_model = SentenceTransformer(model_name)
    else:

        device = "cuda" if torch.cuda.is_available() else "cpu"

        if "bart" in model_name:
            transformer = BartTransformer(model_name, model_args={"output_hidden_states": True})
        elif encoding == "half":
            transformer = HalfPrecisionTransformer(model_name, model_args={"output_hidden_states": True})
        elif encoding == "four_bit":
            transformer = FourBitTransformer(model_name, model_args={"output_hidden_states": True})
        elif encoding == "eight_bit":
            transformer = EightBitTransformer(model_name, model_args={"output_hidden_states": True})
        else:
            transformer = Transformer(model_name, model_args={"output_hidden_states": True})

        if control_task_type == CONTROL_TASK_TYPES.RANDOM_WEIGHTS and model_name in ["t5-base", "roberta-base", "microsoft/deberta-base", "microsoft/deberta-v3-base", "bert-base-uncased", "albert-base-v2", "google/electra-base-discriminator"]:
            transformer.auto_model.encoder.apply(init_random_weights)
        elif control_task_type == CONTROL_TASK_TYPES.RANDOM_WEIGHTS and model_name in ["facebook/bart-base"]:
            transformer.auto_model.encoder.apply(init_random_weights)
            transformer.auto_model.decoder.apply(init_random_weights)
        elif control_task_type == CONTROL_TASK_TYPES.RANDOM_WEIGHTS and model_name in ["gpt2"]:
            transformer.auto_model.h.apply(init_random_weights)

        word_embedding_dimension = transformer.get_word_embedding_dimension()

        pooling = SpecificLayerPooling(
            word_embedding_dimension=word_embedding_dimension,
            layers=[-1], pooling_mode="mean"
        )

        base_model = ParallelSentenceTransformer(modules=[transformer,pooling])

        if device == "cuda" and "cuda" not in str(base_model.device):
            base_model = base_model.to(device)


        if base_model.tokenizer.pad_token is None:
            base_model.tokenizer.pad_token = base_model.tokenizer.eos_token

    return base_model