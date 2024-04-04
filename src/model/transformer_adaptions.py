import torch
from sentence_transformers.models import Transformer
from transformers import AutoModel


class BartTransformer(Transformer):
    def forward(self, features):
        """Returns token_embeddings, cls_token"""
        trans_features = {'input_ids': features['input_ids'], 'attention_mask': features['attention_mask']}
        if 'token_type_ids' in features:
            trans_features['token_type_ids'] = features['token_type_ids']

        output_states = self.auto_model(**trans_features, return_dict=False)
        output_tokens = output_states[0]

        features.update({'token_embeddings': output_tokens, 'attention_mask': features['attention_mask']})

        if self.auto_model.config.output_hidden_states:
            hidden_states = list(output_states[4])[1:] + list(output_states[2])
            features.update({'all_layer_embeddings': hidden_states})

        return features

class SwitchTransformer(Transformer):
    def _load_model(self, model_name_or_path, config, cache_dir):
        from transformers import SwitchTransformersEncoderModel
        SwitchTransformersEncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        self.auto_model = SwitchTransformersEncoderModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)


class HalfSwitchTransformer(Transformer):
    def _load_model(self, model_name_or_path, config, cache_dir):
        from transformers import SwitchTransformersModel
        SwitchTransformersModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        self.auto_model = SwitchTransformersModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir, torch_dtype=torch.float16)

class T5Transformer(Transformer):
    def forward(self, features):
        """Returns token_embeddings, cls_token"""
        trans_features = {'input_ids': features['input_ids'], 'attention_mask': features['attention_mask']}
        if 'token_type_ids' in features:
            trans_features['token_type_ids'] = features['token_type_ids']

        output_states = self.auto_model(**trans_features, return_dict=False)
        output_tokens = output_states[0]

        features.update({'token_embeddings': output_tokens, 'attention_mask': features['attention_mask']})

        if self.auto_model.config.output_hidden_states:
            hidden_states = list(output_states[4])[1:] + list(output_states[2])
            features.update({'all_layer_embeddings': hidden_states})

        return features

    def _load_model(self, model_name_or_path, config, cache_dir):
        self.auto_model = AutoModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)

