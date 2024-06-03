import logging
from collections import defaultdict
from typing import List, Union

import numpy as np
import torch
from numpy import ndarray
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device
from torch import Tensor
from tqdm.autonotebook import trange
from transformers import LlamaModel

logger = logging.getLogger(__name__)

class ParallelSentenceTransformer(SentenceTransformer):

    def encode(self, sentences: Union[str, List[str]],
               batch_size: int = 32,
               show_progress_bar: bool = None,
               output_value: str = 'sentence_embedding',
               convert_to_numpy: bool = True,
               convert_to_tensor: bool = False,
               device: str = None,
               normalize_embeddings: bool = False) -> Union[List[Tensor], ndarray, Tensor]:
        """
        Computes sentence embeddings

        :param sentences: the sentences to embed
        :param batch_size: the batch size used for the computation
        :param show_progress_bar: Output a progress bar when encode sentences
        :param output_value:  Default sentence_embedding, to get sentence embeddings. Can be set to token_embeddings to get wordpiece token embeddings. Set to None, to get all output values
        :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any setting from convert_to_numpy
        :param device: Which torch.device to use for the computation
        :param normalize_embeddings: If set to true, returned vectors will have length 1. In that case, the faster dot-product (util.dot_score) instead of cosine similarity can be used.

        :return:
           By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned. If convert_to_numpy, a numpy matrix is returned.
        """
        self.eval()
        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel()==logging.INFO or logger.getEffectiveLevel()==logging.DEBUG)

        if convert_to_tensor:
            convert_to_numpy = False

        if output_value != 'sentence_embedding' and not output_value != 'token_layer_embeddings':
            convert_to_tensor = False
            convert_to_numpy = False

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(sentences, '__len__'): #Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        #if device is None:
        #    device = self._target_device

        #self.to(device)

        all_embeddings = defaultdict(list)
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        if not self.tokenizer.pad_token :#and type(list(self)[0].auto_model) == GPT2Model:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index:start_index+batch_size]
            features = self.tokenize(sentences_batch)
            features = batch_to_device(features, device)

            if type(self[0].auto_model) == LlamaModel and "token_type_ids" in features:
                del features["token_type_ids"]

            with torch.no_grad():
                out_features = self.forward(features)

                for layer in out_features["sentence_layer_embeddings"].keys():
                    if output_value == 'token_layer_embeddings':
                        embeddings = []
                        for token_emb, attention in zip(out_features[output_value][layer], out_features['attention_mask']):
                            last_mask_id = len(attention)-1
                            while last_mask_id > 0 and attention[last_mask_id].item() == 0:
                                last_mask_id -= 1

                            token_embeddings = token_emb[0:last_mask_id+1].detach().cpu()

                            if normalize_embeddings:
                                token_embeddings = torch.nn.functional.normalize(token_embeddings, p=2, dim=1)


                            embeddings.append(token_embeddings)
                    elif output_value is None:  #Return all outputs
                        embeddings = []
                        for sent_idx in range(len(out_features['sentence_layer_embeddings'][layer])):
                            row =  {name: out_features[name][sent_idx].detach().cpu() for name in out_features}
                            embeddings.append(row)
                    else:   #Sentence embeddings
                        embeddings = out_features['sentence_layer_embeddings'][layer]
                        embeddings = embeddings.detach().cpu()
                        if normalize_embeddings:
                            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                        # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                        if convert_to_numpy:
                            embeddings = embeddings.cpu()

                    all_embeddings[layer].extend(embeddings)

        for layer in all_embeddings.keys():

            all_embeddings[layer] = [all_embeddings[layer][idx] for idx in np.argsort(length_sorted_idx)]

            if convert_to_tensor:
                all_embeddings[layer] = torch.stack(all_embeddings[layer])
            elif convert_to_numpy:
                all_embeddings[layer] = np.asarray([emb.numpy() for emb in all_embeddings[layer]])

            if input_was_string:
                all_embeddings[layer] = all_embeddings[layer][0]

        return all_embeddings


    def encode_features(self, sentences: Union[str, List[str]],
               batch_size: int = 32,
               show_progress_bar: bool = None,
               device: str = None,
               normalize_embeddings: bool = False) -> Union[List[Tensor], ndarray, Tensor]:
        self.eval()
        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel()==logging.INFO or logger.getEffectiveLevel()==logging.DEBUG)


        if isinstance(sentences, str) or not hasattr(sentences, '__len__'): #Cast an individual sentence to a list with length 1
            sentences = [sentences]

        #if device is None:
        #    device = self._target_device

        self.to(device)

        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        all_out_features = {}

        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index:start_index+batch_size]
            features = self.tokenize(sentences_batch)
            features = batch_to_device(features, device)

            out_features = self.forward(features)

            for i, sentence in enumerate(sentences_batch):
                length = int((out_features["input_ids"][0] == 0).sum())
                all_out_features[sentence] = {
                    "token_embeddings": out_features["token_embeddings"][i, -length],
                    "all_layer_embeddings": [features[i, -length] for features in out_features["all_layer_embeddings"]],
                    "sentence_layer_embeddings": {layer: features[i, -length] for layer, features in out_features["sentence_layer_embeddings"].items()},
                    "token_layer_embeddings": {layer: features[i, -length] for layer, features in out_features["token_layer_embeddings"].items()},
                }


        return all_out_features

