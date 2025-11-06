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
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from definitions.control_task_types import CONTROL_TASK_TYPES
from definitions.probe_task_types import PROBE_TASK_TYPES
from definitions.schema import ProbingEntry, ProbingTask, ScalarProbingDataset, ScalarProbingEntry


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
