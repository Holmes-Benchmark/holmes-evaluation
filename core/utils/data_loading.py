import gc
import os.path
import pickle
import random
from pathlib import Path
from typing import List, Dict

import gc
import itertools
import numpy
import pandas
import ray
import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset
from tqdm import tqdm

from defs.control_task_types import CONTROL_TASK_TYPES
from defs.probe_task_types import PROBE_TASK_TYPES
from defs.schema import ProbingEntry, ProbingTask, ScalarProbingDataset, ScalarProbingEntry
from utils.model_loading import load_model


def get_unique_inputs(inputs):
    return set([tuple([ele[0].lower() for ele in entry]) for entry in inputs])

class ProbingDataset(Dataset):
    def __init__(self, inputs, inputs_encoded, labels):
        self.inputs = inputs
        self.unique_inputs = get_unique_inputs(self.inputs)
        self.labels = labels
        self.seen = [False for ele in self.inputs]
        self.inputs_encoded = inputs_encoded

    def __getitem__(self, index):
        return self.seen[index], self.inputs_encoded[index], self.labels[index]

    def __len__(self):
        return len(self.inputs_encoded)

    def get_seen_indices(self):
        return [i for i, ele in enumerate(self.seen) if ele]

    def get_unseen_indices(self):
        return [i for i, ele in enumerate(self.seen) if not ele]

    def update_seen(self, ref_unique_inputs):
        self.seen = [True if tuple([ele[0].lower() for ele in element]) in ref_unique_inputs else False for element in self.inputs]


class SharedProbingDataset(Dataset):
    def __init__(self, inputs, inputs_encoded, labels):
        self.inputs = inputs
        self.unique_inputs = get_unique_inputs(self.inputs)
        self.labels = labels
        self.seen = [False for ele in self.inputs]

        self.inputs_encoded = ray.put(inputs_encoded)

    def __getitem__(self, index):
        return self.seen[index], self.inputs_encoded[index], self.labels[index]

    def __len__(self):
        return len(self.inputs_encoded)

    def get_seen_indices(self):
        return [i for i, ele in enumerate(self.seen) if ele]

    def get_unseen_indices(self):
        return [i for i, ele in enumerate(self.seen) if not ele]

    def update_seen(self, ref_unique_inputs):
        self.seen = [True if tuple([ele[0].lower() for ele in element]) in ref_unique_inputs else False for element in self.inputs]

    def attach_memory(self):
        self.inputs_encoded = ray.get(self.inputs_encoded)

def parse_entries(entities_frame: pandas.DataFrame, test=False):
    return [
        ProbingEntry(
            id=row["id"],
            inputs=row["inputs"] if test else "",
            inputs_encoded=row["inputs_encoded"],
            context=row["context"] if test else "",
            label=row["label"],
        )
        for index, row in entities_frame.iterrows()
    ]

def parse_scalar_entries(entities_frame: pandas.DataFrame, test=False):
    return [
        ScalarProbingEntry(
            id=row["id"],
            inputs=row["inputs"],
            inputs_encoded=row["inputs_encoded"],
            context=row["context"] if test else "",
            label=row["label"],
        )
        for index, row in entities_frame.iterrows()
    ]

def parse_fold_frame(fold_frame:pandas.DataFrame):

    return ProbingDataset(
        train_entries=parse_entries(fold_frame[fold_frame["set"] == "train"]),
        dev_entries=parse_entries(fold_frame[fold_frame["set"] == "dev"]),
        test_entries=parse_entries(fold_frame[fold_frame["set"] == "test"], test=True),
    )

def parse_scalar_fold_frame(fold_frame:pandas.DataFrame):

    return ScalarProbingDataset(
        train_entries=parse_scalar_entries(fold_frame[fold_frame["set"] == "train"]),
        dev_entries=parse_scalar_entries(fold_frame[fold_frame["set"] == "dev"]),
        test_entries=parse_scalar_entries(fold_frame[fold_frame["set"] == "test"], test=True),
    )


def parse_probe_folds(folds_frames:Dict[int, pandas.DataFrame])->ProbingTask:

    folds = {
        fold:parse_fold_frame(fold_frame)
        for fold, fold_frame in folds_frames.items()
    }


    probing_task = ProbingTask(
        folds=folds
    )

    return probing_task

def default_collate(element, encoding):

    encoded_inputs = numpy.stack([numpy.array(layers).flatten() for layers in element])

    if encoding == "half" or encoding == "four_bit":
        encoded_inputs = encoded_inputs.astype(numpy.float16)

    return encoded_inputs

def scalar_mix_collate(element, encoding):

    encoded_inputs = numpy.stack([numpy.array(layers).flatten() for layers in element])

    if encoding == "half" or encoding == "four_bit":
        encoded_inputs = encoded_inputs.astype(numpy.float16)

    return encoded_inputs

def permutate_words(sentence):
    words = sentence.split(" ")

    random.shuffle(words)

    return " ".join(words)


def permutate_context(row):
    all_inputs = [ele[0] for ele in row["inputs"]]
    updated_inputs = []

    permutated_context = []
    context = eval(row["context"])
    for i, context_element in enumerate(context):
        context_element = context_element
        observed_input_elements = []
        for input_str, input_ele in sorted(zip(all_inputs, row["inputs"]), key=lambda ele: len(ele[0]), reverse=True):
            if input_str in context_element and i == input_ele[1]:
                num_occurrences = context_element.split().count(input_str)
                if num_occurrences == 0:
                    num_occurrences = context_element.count(input_str)

                observed_input_elements += [input_str] * num_occurrences
                context_element = context_element.replace(input_str, " ")

        context_element = context_element.replace("  ", " ")
        context_elements = context_element.split(" ") + observed_input_elements

        random.Random(0).shuffle(context_elements)

        permutated_context.append(" ".join(context_elements))

        for input_str, input_ele in zip(all_inputs, row["inputs"]):
            if i == input_ele[1]:
                occurrences = [(j, ele) for j, ele in enumerate(context_elements) if ele == input_str]
                new_occurrence = random.Random(0).choice(occurrences)

                start_index = 0

                for j, ele in enumerate(context_elements):
                    if j == new_occurrence[0]:
                        updated_inputs.append((
                            input_str, i, start_index, start_index + len(input_str)
                        ))
                        break

                    start_index += len(ele) + 1



                observed_input_elements += [input_str] * num_occurrences
                context_element = context_element.replace(input_str, " ")

    row["context"] = str(permutated_context)
    row["inputs"] = updated_inputs

    return row


def process_frame(frame:pandas.DataFrame, control_task_type:CONTROL_TASK_TYPES):
    if control_task_type == CONTROL_TASK_TYPES.PERMUTATION:
        if list(frame["context"])[0] != "":
            frame = frame.apply(lambda row: permutate_context(row), axis=1)
        else:
            frame.loc[:,"inputs"] = frame["inputs"].apply(lambda ele: [permutate_words(input_str) for input_str in ele])
        return frame
    elif control_task_type == CONTROL_TASK_TYPES.RANDOMIZATION:
        random_labels = numpy.random.permutation(frame["label"].values)
        frame["label"] = random_labels

        return frame
    else:
        return frame

def load_probe_file(probe_file:str, control_task_type:CONTROL_TASK_TYPES, sample_size=0):
    loaded_frame = pandas.read_csv(probe_file).sort_values("id")

    if sample_size > 0 and sample_size < loaded_frame.shape[0]:
        loaded_frame = pandas.concat([
            loaded_frame[loaded_frame["set-0"] == "train"].sample(sample_size),
            loaded_frame[loaded_frame["set-0"] == "dev"],
            loaded_frame[loaded_frame["set-0"] == "test"],
        ])

    if not "id" in loaded_frame.columns:
        loaded_frame["id"] = loaded_frame.index

    if str(loaded_frame["context"].values[0]) == "nan":
        loaded_frame.loc[:,"context"] = ""
    loaded_frame.loc[:,"inputs"] = loaded_frame["inputs"].apply(lambda ele: eval(ele))
    processed_frame = process_frame(loaded_frame, control_task_type)
    return processed_frame

def compare(string1, string2, no_match_c=' ', match_c='|'):
    if len(string2) < len(string1):
        string1, string2 = string2, string1
    result = ''
    n_diff = 0
    for c1, c2 in itertools.izip(string1, string2):
        if c1 == c2:
            result += match_c
        else:
            result += no_match_c
            n_diff += 1
    delta = len(string2) - len(string1)
    result += delta * no_match_c
    n_diff += delta
    return n_diff

def find_sub_list(input_element_start, input_element_end, context_tokenized):
    input_indices = []

    for char_index in range(input_element_start, input_element_end):

        token_index = context_tokenized.char_to_token(char_index)

        if token_index is not None and token_index not in input_indices:
            input_indices.append(token_index)

    return input_indices

def normalize(input_string, base_model):
    return base_model.tokenizer.backend_tokenizer.normalizer.normalize_str(input_string)

def pre_tokenize(input_string, base_model):
    input_string = input_string.lower()
    pre_tokenized_string = base_model.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(input_string)

    return pre_tokenized_string

def find_sub_list_start(sub_list,l):
    results=[]
    sll=len(sub_list)
    for ind in (i for i,e in enumerate(l) if e==sub_list[0]):
        if l[ind:ind+sll]==sub_list:
            results.append(ind)

    return results

def extract_embeddings_for_input_element(input_element:str, context:str, pre_tokenized_context, layer_embeddings, base_model:SentenceTransformer, probe_task_type:PROBE_TASK_TYPES, model_type:str, encoding:str):
    input_token, input_element_index, input_token_start, input_token_end = input_element

    if not hasattr(base_model.tokenizer, "sep_token") or base_model.tokenizer.sep_token is None:
        sep_token = ""
    else:
        sep_token = base_model.tokenizer.sep_token


    device = "cuda" if torch.cuda.is_available() else "cpu"

    if probe_task_type in [PROBE_TASK_TYPES.SENTENCE_TOKENS, PROBE_TASK_TYPES.SENTENCE_SPANS, PROBE_TASK_TYPES.SENTENCE_PAIR_TOKENS_BI]:
        if model_type == "glove":
            input_token_tokenized = base_model.tokenizer.tokenize(input_token)
            if len(input_token_tokenized) == 0:
                return base_model.encode(["unk"], device=device)
            input_indices = find_sub_list_start(input_token_tokenized, pre_tokenized_context[input_element_index])
            if len(input_indices) > 0:
                input_indices = range(input_indices[0], input_indices[0] + len(input_token_tokenized))
            else:
                return base_model.encode(["unk"], device=device)
        else:
            try:
                input_indices = find_sub_list(input_token_start, input_token_end, pre_tokenized_context[input_element_index])
            except:
                print(pre_tokenized_context[input_element_index])

        embeddings = torch.stack([context_embeddings[context[input_element_index]] for layer, context_embeddings in layer_embeddings.items()])

    else:
        first_context, second_context = context

        joined_context = (" " + sep_token + " ").join(context)

        if input_element_index == 0:
            input_indices = find_sub_list(input_token_start, input_token_end, pre_tokenized_context)
        else:
            input_indices = find_sub_list(input_token_start + len(first_context) + 7, input_token_end + len(first_context) + 7, pre_tokenized_context)

        if not input_indices and model_type == "glove":
            return base_model.encode(["unk"], device=device)

        embeddings = torch.stack([context_embeddings[joined_context] for layer, context_embeddings in layer_embeddings.items()])

    try:
        selected_embeddings = embeddings[:, input_indices].mean(dim=1).cpu().detach().numpy()
    except:
        print()

    if encoding == "half" or encoding == "four_bit":
        selected_embeddings = selected_embeddings.astype(numpy.float16)

    return selected_embeddings




def encode_inputs(inputs:List[List[str]], context:List[str], base_model:SentenceTransformer, probe_task_type:PROBE_TASK_TYPES, encoding_batch_size=10, encoding="full"):
    try:
        model_type = type(base_model[0].auto_model).__name__
        output_value = "token_layer_embeddings"
    except:
        model_type = "glove"
        output_value = "token_embeddings"

    if not hasattr(base_model.tokenizer, "sep_token") or base_model.tokenizer.sep_token is None:
        sep_token = ""
    else:
        sep_token = base_model.tokenizer.sep_token

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if probe_task_type in [PROBE_TASK_TYPES.SENTENCE_TOKENS, PROBE_TASK_TYPES.SENTENCE_PAIR_TOKENS_BI, PROBE_TASK_TYPES.SENTENCE_PAIR_TOKENS_CROSS]:

        if probe_task_type in [PROBE_TASK_TYPES.SENTENCE_TOKENS, PROBE_TASK_TYPES.SENTENCE_PAIR_TOKENS_BI]:
            reduced_context = list(set(itertools.chain(*[eval(ele) for ele in context])))
        else:
            reduced_context = list(set([(" " + sep_token + " ").join(eval(ele)) for ele in context]))


        encoded_context = base_model.encode(
            sentences=reduced_context, show_progress_bar=True, batch_size=encoding_batch_size,
            output_value=output_value, convert_to_numpy=True, device=device
        )

        if model_type == "glove":
            encoded_context = {0: encoded_context}

        encoded_context_dict = {
            layer:dict(zip(reduced_context, layer_encoded_inputs))
            for layer, layer_encoded_inputs in encoded_context.items()
        }

        del encoded_context

        context = [
            eval(entry)
            for entry in context
        ]

        if model_type == "glove":
            pre_tokenized_context = [
                [base_model.tokenizer.tokenize(ele) for ele in entry]
                for entry in tqdm(context)
            ]
        else:
            pre_tokenized_context = [
                [base_model.tokenizer(ele) for ele in entry]
                for entry in tqdm(context)
            ]


        encoded_inputs = [
                [
                    extract_embeddings_for_input_element(
                        input_element, context_ele, pre_tokenized_context_ele, encoded_context_dict, base_model, probe_task_type, model_type, encoding
                    )
                    for input_element in input_list
                ]
            for input_list, context_ele, pre_tokenized_context_ele in tqdm(zip(inputs, context, pre_tokenized_context))
        ]

        del context
        del pre_tokenized_context

        return {
            layer: [
                [input_element[i] for input_element in encoded_input]
                for encoded_input in encoded_inputs
            ] for i, layer in enumerate(encoded_context_dict.keys())
        }
    elif probe_task_type in [PROBE_TASK_TYPES.SENTENCE_SPANS]:

        if probe_task_type in [PROBE_TASK_TYPES.SENTENCE_SPANS]:
            reduced_context = list(set(itertools.chain(*[eval(ele) for ele in context])))
        else:
            reduced_context = list(set([(" " + sep_token + " ").join(eval(ele)) for ele in context]))

        encoded_context = base_model.encode(
            sentences=reduced_context, show_progress_bar=True, batch_size=encoding_batch_size,
            output_value=output_value, convert_to_numpy=True, device=device
        )

        if model_type == "glove":
            encoded_context = {0: encoded_context}

        encoded_context_dict = {
            layer:dict(zip(reduced_context, layer_encoded_inputs))
            for layer, layer_encoded_inputs in encoded_context.items()
        }

        del encoded_context

        context = [
            eval(entry)
            for entry in context
        ]

        if model_type == "glove":
            pre_tokenized_context = [
                [base_model.tokenizer.tokenize(ele) for ele in entry]
                for entry in tqdm(context)
            ]
        else:
            pre_tokenized_context = [
                [base_model.tokenizer(ele) for ele in entry]
                for entry in tqdm(context)
            ]

        n_inputs = len(set([ele[-1] for ele in inputs[0]]))

        dimension_encodings = []

        for input_index in range(n_inputs):
            filtered_input = [
                tuple([input_element[:-1] for input_element in input_list if input_element[-1] == input_index])
                for input_list in inputs
            ]

            encoded_filtered_inputs = [
                [
                    extract_embeddings_for_input_element(
                        input_element, context_ele, pre_tokenized_context_ele, encoded_context_dict, base_model, probe_task_type, model_type, encoding
                    )
                    for input_element in input_list
                ]
                for input_list, context_ele, pre_tokenized_context_ele in tqdm(zip(filtered_input, context, pre_tokenized_context))
            ]

            dimension_encodings.append([numpy.array(ele).mean(axis=0) for ele in encoded_filtered_inputs])

        del context
        del pre_tokenized_context

        return {
            layer: [
                [dimension_encodings[input_index][j][i] for input_index in range(n_inputs)]
                for j in range(len(encoded_filtered_inputs))
            ] for i, layer in enumerate(encoded_context_dict.keys())
        }

    else:

        if probe_task_type in [PROBE_TASK_TYPES.SENTENCE, PROBE_TASK_TYPES.SENTENCE_PAIR_BI]:

            flatten_inputs = list(set(itertools.chain(*inputs)))

            encoded_inputs = base_model.encode(
                sentences=flatten_inputs, show_progress_bar=True, batch_size=encoding_batch_size, device=device
            )

            if model_type == "glove":
                encoded_inputs = {0: encoded_inputs}

            encoded_inputs_dict = {
                layer:dict(zip(flatten_inputs, layer_encoded_inputs))
                for layer, layer_encoded_inputs in encoded_inputs.items()
            }

            if encoding == "half" or encoding == "four_bit":
                dtype = numpy.float
            else:
                dtype = numpy.float16

            return {
                layer: [numpy.array([layer_encoded_inputs_dict[ele] for ele in input]).astype(dtype) for input in inputs]
                for layer, layer_encoded_inputs_dict in encoded_inputs_dict.items()
            }

        elif probe_task_type in [PROBE_TASK_TYPES.SENTENCE_PAIR_CROSS]:

            flatten_inputs = [(" " + sep_token + " ").join(ele) for ele in inputs]
            encoded_inputs = base_model.encode(
                sentences=flatten_inputs, show_progress_bar=True, batch_size=encoding_batch_size, device=device
            )

            if model_type == "glove":
                encoded_inputs = {0: encoded_inputs}

            if encoding == "half" or encoding == "four_bit":
                dtype = numpy.float
            else:
                dtype = numpy.float16

            return {
                layer: [numpy.array(ele).astype(dtype) for ele in encoded_layer_inputs.tolist()]
                for layer, encoded_layer_inputs in encoded_inputs.items()
            }


def load_probing_frames(probing_frames, encoding):
    loaded_frames = []

    for fold, probing_frame in enumerate(probing_frames):
        last_key = list(probing_frame.keys())[-1]

        joined_frame = list(probing_frame.values())[0].copy()
        joined_frame["inputs_encoded"] = [
            probing_frame[last_key].loc[index,"inputs_encoded"]
            for index, row in tqdm(joined_frame.iterrows())
        ]
        joined_frame["inputs_encoded"] = joined_frame["inputs_encoded"].apply(lambda ele: default_collate(ele, encoding))
        joined_frame["inputs_encoded"] = joined_frame["inputs_encoded"].apply(lambda ele: ele.flatten())
        joined_frame["unique_inputs"] = joined_frame["inputs"].apply(lambda element: tuple([ele[0].lower() for ele in element]))

        loaded_frames.append({
            "train": joined_frame[joined_frame["set-" + str(fold)] == "train"],
            "dev": joined_frame[joined_frame["set-" + str(fold)] == "dev"],
            "test": joined_frame[joined_frame["set-" + str(fold)] == "test"],
        })

    return loaded_frames


def load_dataset(probing_frame):
    inputs = probing_frame["inputs"].values
    labels = probing_frame["label"].values
    inputs_encoded = probing_frame["inputs_encoded"].values

    return ProbingDataset(inputs, inputs_encoded, labels)


def load_datasets(probing_frames):
    train_probing_frame = probing_frames["train"]
    dev_probing_frame = probing_frames["dev"]
    test_probing_frame = probing_frames["test"]

    train_dataset = load_dataset(train_probing_frame)
    dev_dataset = load_dataset(dev_probing_frame)
    test_dataset = load_dataset(test_probing_frame)

    dev_dataset.update_seen(train_dataset.unique_inputs)
    test_dataset.update_seen(train_dataset.unique_inputs)

    return train_dataset, dev_dataset, test_dataset



def load_shared_dataset(probing_frame):
    inputs = probing_frame["inputs"].values
    labels = probing_frame["label"].values
    inputs_encoded = probing_frame["inputs_encoded"].values

    return SharedProbingDataset(inputs, inputs_encoded, labels)


def load_shared_datasets(probing_frames):
    train_probing_frame = probing_frames["train"]
    dev_probing_frame = probing_frames["dev"]
    test_probing_frame = probing_frames["test"]

    train_dataset = load_shared_dataset(train_probing_frame)
    dev_dataset = load_shared_dataset(dev_probing_frame)
    test_dataset = load_shared_dataset(test_probing_frame)

    dev_dataset.update_seen(train_dataset.unique_inputs)
    test_dataset.update_seen(train_dataset.unique_inputs)

    return train_dataset, dev_dataset, test_dataset

def dump_data(probe_frame, probe_task_type, control_task_type, encoding_batch_size, dump_path, model_name, encoding, scalar_mixin=False):
    base_model = load_model(model_name, control_task_type, encoding, scalar_mixin=scalar_mixin)

    probing_frames = load_folds(
        probe_frame=probe_frame,
        base_model=base_model,
        probe_task_type=probe_task_type,
        encoding=encoding,
        encoding_batch_size=encoding_batch_size,
    )

    try:
        pickle.dump(probing_frames, open(f"{dump_path}", "wb"))
    except:
        raise Exception(f"Failed to dump at {dump_path}")


def load_data(dump_folder, dump_id, encoding, scalar_mixin=False):

    dump_id = dump_id.replace('/', "__")

    dump_file = f"{dump_folder}/{dump_id}.pickle"

    if os.path.exists(dump_file):
        probing_frames = pickle.load(open(dump_file, "rb"))
    else:
        raise Exception(f"Dump data not found {dump_file}")

    if scalar_mixin:
        probing_frames = load_scalar_mix_probing_frames(probing_frames, encoding)
    else:
        probing_frames = load_probing_frames(probing_frames, encoding)

    return probing_frames

def load_scalar_mix_probing_frames(probing_frames, encoding):
    loaded_frames = []

    for fold, probing_frame in enumerate(probing_frames):
        joined_frame = list(probing_frame.values())[0].copy()
        joined_frame["inputs_encoded"] = [
            [probing_frame[layer].loc[index,"inputs_encoded"] for layer in probing_frame.keys()]
            for index, row in tqdm(joined_frame.iterrows())
        ]
        joined_frame["inputs_encoded"] = joined_frame["inputs_encoded"].apply(lambda ele: scalar_mix_collate(ele, encoding))
        joined_frame["unique_inputs"] = joined_frame["inputs"].apply(lambda element: tuple([ele[0].lower() for ele in element]))

        loaded_frames.append({
            "train": joined_frame[joined_frame["set-" + str(fold)] == "train"],
            "dev": joined_frame[joined_frame["set-" + str(fold)] == "dev"],
            "test": joined_frame[joined_frame["set-" + str(fold)] == "test"],
        })

    return loaded_frames


def encode_fold_inputs(probe_frame:pandas.DataFrame, probe_task_type:PROBE_TASK_TYPES, base_model:SentenceTransformer, encoding_batch_size=10, encoding="full"):

    inputs_encoded = []

    for frame in tqdm(numpy.array_split(probe_frame, 5)):

        chunk = encode_inputs(
            inputs=list(frame["inputs"]),
            context=list(frame["context"]),
            base_model=base_model,
            probe_task_type=probe_task_type,
            encoding=encoding,
            encoding_batch_size=encoding_batch_size
        )

        inputs_encoded.append(chunk)
        gc.collect()

    inputs_encoded = {
        layer: list(itertools.chain.from_iterable([chunk[layer] for chunk in inputs_encoded]))
        for layer in inputs_encoded[0].keys()
    }

    fold_frames = {}
    probe_frame["context"] = ""
    for layer, layer_inputs_encoded in inputs_encoded.items():
        layer_fold_frame = probe_frame.copy()
        layer_fold_frame["inputs_encoded"] = layer_inputs_encoded
        fold_frames[layer] = layer_fold_frame

    return fold_frames

def load_folds(probe_frame:pandas.DataFrame, base_model:SentenceTransformer, probe_task_type:PROBE_TASK_TYPES, encoding_batch_size=10, encoding="full"):
    encoded_folds = []
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    encoded_folds.append(
        encode_fold_inputs(
            probe_frame=probe_frame,
            base_model=base_model,
            probe_task_type=probe_task_type,
            encoding=encoding,
            encoding_batch_size=encoding_batch_size
        )
    )
    return encoded_folds
