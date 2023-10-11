"""Implements different tasks and defines the processors to convert each dataset
to a sequence to sequence format."""
from collections import OrderedDict

import abc
import datasets
import functools
import logging
import numpy as np
import torch
from metrics import metrics
from typing import Callable, Dict, Mapping, List
from collections import defaultdict
from datasets import load_from_disk
from transformers import T5TokenizerFast
from datasets import concatenate_datasets
import re


from .utils import round_stsb_target, compute_task_max_decoding_length
from .mrqa_preprocess import chunk_sample

logger = logging.getLogger(__name__)


class AbstractTaskDataset(abc.ABC):
    """Defines the abstract class for all the tasks.
    name: the name of the task.
    task_specific_config: specifies the special configuration needs
        to be passed to encoder when decoding each task. Since different
        tasks, have different output space, the maximum decoding length
        varies based on the tasks.
    preprocessor: a processor to convert the given dataset to the sequence
        to sequence format.
    metrics: specifies the metrics to evaluate the task based on them.
    split_to_data_split: since not all the time, different splits of the
        datasets are available, we define a mapping from the wanted split
        to the existing dataset splits.
    small_datasets_without_all_splits: List of strings, defines the name
        of all low-resource tasks in which not all train/test/validation
        splits are available.
    large_data_without_all_splits: List of strings, defines the name of
        all high-resource tasks in which not all train/test/validation
        splits are available.
    """

    name = NotImplemented
    task_specific_config: Dict = NotImplemented
    preprocessor: Callable = NotImplemented
    metrics: List[Callable] = NotImplemented
    split_to_data_split: Mapping[str, str] = {
        "train": "train",
        "validation": "validation",
        "test": "test",
    }

    small_datasets_without_all_splits = ["cola", "wnli", "rte", "superglue-cb", "superglue-copa", "superglue-multirc",
                                         "superglue-wic", "superglue-wsc.fixed", "superglue-rte", "mrpc", "stsb",
                                         "superglue-boolq", "xsum", "scitail"]
    large_data_without_all_splits = ["qqp", "qnli", "superglue-record", "sst2", "squad", "snli", "anli",
                                     "amazon_polarity", "yelp_polarity", "winogrande", "newsqa", "searchqa", "triviaqa", "nq", "hotpotqa"]
    generation_task: bool = False  # for loss scaling

    def __init__(self, seed=42):
        self.seed = seed

    def get_sampled_split(self, split: int, n_obs: int = None):
        # If the requested number of observation is more than dataset
        # size we reset it to the maximum available.
        split = self.split_to_data_split[split]
        dataset = self.load_dataset(split)
        total_size = len(dataset)
        n_obs = self.check_n_obs(n_obs, total_size)
        if n_obs is not None:
            split = split + "[:{}]".format(n_obs)
        return split

    def get_shuffled_sampled_split(self, split: int, n_obs: int = None):
        # Defines the random generator.
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        # If the requested number of observation is more than dataset
        # size we reset it to the maximum available.
        mapped_split = self.split_to_data_split[split]
        dataset = self.load_dataset(mapped_split)
        # shuffle the dataset and get the random samples.
        train_size = len(dataset)
        indices = torch.randperm(train_size, generator=generator).tolist()
        dataset = self.select_dataset_samples(indices, dataset, n_obs=n_obs)
        return dataset

    def check_n_obs(self, n_obs, total_size):
        if n_obs is not None and n_obs > total_size:
            n_obs = total_size
            logger.warning("n_obs is set to %s", n_obs)
        return n_obs

    def select_dataset_samples(self, indices, dataset, n_obs: int = None):
        """
        Given a dataset for the split, obtains the sample indices for this split
        and returns the subsampled dataset.
        :param indices: the selected indices.
        :param dataset: dataset corresponding to this split.
        :return: subsampled dataset.
        """
        n_obs = self.check_n_obs(n_obs, len(indices))
        indices = indices[:n_obs] if n_obs is not None else indices
        return dataset.select(indices)

    def load_dataset(self, split: int):
        return datasets.load_dataset(self.name, split=split)

    def get_train_split_indices(self, split):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        mapped_split = self.split_to_data_split["train"]
        dataset = self.load_dataset(mapped_split)
        train_size = len(dataset)
        indices = torch.randperm(train_size, generator=generator).tolist()
        validation_size = 1000
        if split == "validation":
            return indices[:validation_size]
        else:
            return indices[validation_size:]

    def get_half_validation_indices(self, split):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        mapped_split = self.split_to_data_split["validation"]
        dataset = self.load_dataset(mapped_split)
        validation_size = len(dataset)
        indices = torch.randperm(validation_size, generator=generator).tolist()
        if split == "validation":
            return indices[: (validation_size // 2)]
        else:
            return indices[validation_size // 2 :]

    def get_dataset(
        self, split, n_obs=None, add_prefix=True, split_validation_test=False
    ):
        # For small datasets (n_samples < 10K) without test set, we divide validation set to
        # half, use one half as test set and one half as validation set.
        if (
            split_validation_test
            and self.name in self.small_datasets_without_all_splits
            and split != "train"
        ):
            mapped_split = self.split_to_data_split["validation"]
            dataset = self.load_dataset(split=mapped_split)
            indices = self.get_half_validation_indices(split)
            dataset = self.select_dataset_samples(indices, dataset, n_obs)
        # For larger datasets (n_samples > 10K), we divide training set into 1K as
        # validation and the rest as training set, keeping the original validation
        # set as the test set.
        elif (
            split_validation_test
            and self.name in self.large_data_without_all_splits
            and split != "test"
        ):
            dataset = self.load_dataset(split="train")
            indices = self.get_train_split_indices(split)
            dataset = self.select_dataset_samples(indices, dataset, n_obs)
        else:
            dataset = self.get_shuffled_sampled_split(split, n_obs)
        return dataset.map(
            functools.partial(self.preprocessor, add_prefix=add_prefix),
            remove_columns=dataset.column_names,
        )

    def seq2seq_format(
        self,
        src_strs: List[str],
        tgt_strs: List[str],
        add_prefix: bool = False,
        prefix: str = None,
        id: str = None,
    ):
        src_prefix = self.name if prefix is None else prefix
        src_strs = [src_prefix] + src_strs if add_prefix else src_strs
        return {
            "src_texts": " ".join(src_strs),
            "tgt_texts": " ".join(tgt_strs),
            "task": self.name,
            "id": id,  # for squad, we need to save id: answer mapping.
        }

    def get_label_size(self, tokenizer):
        if self.generation_task:
            return tokenizer.vocab_size
        else:
            return len(self.label_list)


class IMDBTaskDataset(AbstractTaskDataset):
    name = "imdb"
    split_to_data_split = {"train": "train", "validation": "test", "test": "test"}
    label_list = ["0", "1"]
    task_specific_config = {"max_length": compute_task_max_decoding_length(label_list)}
    metrics = [metrics.accuracy]

    def preprocessor(self, example, add_prefix=True):
        src_texts = [example["text"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SickTaskDataset(AbstractTaskDataset):
    name = "sick"
    label_list = ["0", "1", "2"]
    task_specific_config = {"max_length": compute_task_max_decoding_length(label_list)}
    split_to_data_split = {"train": "train", "validation": "validation", "test": "test"}
    label_to_target = {"ENTAILMENT": 0, "CONTRADICTION": 2, "NEUTRAL": 1}
    metrics = [metrics.accuracy]

    def load_dataset(self, split: int):
        return datasets.load_dataset(
            "csv", data_files={split: f"sick/{split}_clean.csv"}
        )[split]

    def preprocessor(self, example, add_prefix=True):
        src_texts = [
            "premise:",
            example["premise"],
            "hypothesis:",
            example["hypothesis"],
        ]
        tgt_texts = [str(self.label_to_target[example["label"]])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class PawsTaskDataset(AbstractTaskDataset):
    name = "paws"
    label_list = ["0", "1"]
    task_specific_config = {"max_length": compute_task_max_decoding_length(label_list)}
    split_to_data_split = {"train": "train", "validation": "validation", "test": "test"}
    metrics = [metrics.accuracy]

    def load_dataset(self, split: int):
        return datasets.load_dataset(
            self.name, "labeled_final", split=split
        )

    def preprocessor(self, example, add_prefix=True):
        src_texts = [
            "sentence1:",
            example["sentence1"],
            "sentence2:",
            example["sentence2"],
        ]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)
    
    
class SuperGLUEMultiRC(AbstractTaskDataset):
    name = "superglue-multirc"
    label_list = ['0', '1']
    task_specific_config = {"max_length": compute_task_max_decoding_length(label_list)}
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metrics = [metrics.f1_score_with_invalid]

    def load_dataset(self, split):
        temp = load_from_disk("/root/datasets/multirc")
        
        return temp[split]

    def remove_markup(self, text):
        """Removes the HTML markup."""
        text = re.sub('<br>', ' ', text)
        text = re.sub('<(/)?b>', '', text)
        return text

    def preprocessor(self, example, add_prefix=True):
        # T5 applies remove_markup to the joined string, but this should not make
        # any difference as well.
        # https://github.com/google-research/text-to-text-transfer-transformer/blob/a1352e625db7ec114062f99d99b0565b9e45c155/t5/data/preprocessors.py#L797
        src_texts = ["question:", self.remove_markup(example["question"]),
                     "answer:", self.remove_markup(example["answer"]),
                     "paragraph:", self.remove_markup(example["paragraph"])]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)    

    
class SuperGLUEWIC(AbstractTaskDataset):
    name = "superglue-wic"
    label_list = ['0', '1']
    task_specific_config = {"max_length": compute_task_max_decoding_length(label_list)}
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metrics = [metrics.accuracy]

    def load_dataset(self, split):
        temp = load_from_disk("/root/datasets/wic")
        
        return temp[split]

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example["sentence1"],
                     "sentence2:", example["sentence2"],
                     "word:", example["word"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)

class SuperGLUEWSCFixed(AbstractTaskDataset):
    # source: https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py
    """Convert WSC examples to text2text format.
     WSC includes a sentence along with 2 'spans': the first denoting a noun and
     the other a pronoun. The 'label' specifies whether or not the pronoun is
     referencing the noun. This preprocessor puts ' * ' around the noun and ' # '
     around the pronoun.
     For example, a typical example from WSC might look like
     {
         'text': 'This is a test sentence .',
         'span1_text': 'test',
         'span1_index': 3,
         'span2_text': 'This',
         'span2_index': 0,
         'label': 0
     }
     This example would be transformed to
     {
         'inputs': 'wsc text: # This # is a * test * sentence .',
         'targets': 'False'
     }
    """
    name = "superglue-wsc.fixed"
    label_list = ['0', '1']
    task_specific_config = {"max_length": compute_task_max_decoding_length(label_list)}
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metrics = [metrics.accuracy]


    def load_dataset(self, split):
        temp = load_from_disk("/root/datasets/wsc")
        
        return temp[split]
    def _mark_span(self, text, span_str, span_idx, mark):
        pattern_tmpl = r'^((?:\S+\s){N})(W)'
        pattern = re.sub('N', str(span_idx), pattern_tmpl)
        pattern = re.sub('W', span_str, pattern)
        return re.sub(pattern, r'\1{0} \2 {0}'.format(mark), text)

    def preprocessor(self, example, add_prefix=True):
        # converts text as done in T5.
        text = example['text']
        text = self._mark_span(
            text, example['span1_text'], example['span1_index'], '*')
        # Compensate for 2 added "words" added in previous step.
        span2_index = example['span2_index'] + 2 * \
            int(example['span1_index'] < example['span2_index'])
        text = self._mark_span(text, example['span2_text'], span2_index, '#')
        src_texts = ["text:", text]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)
    
class SuperGLUECOPA(AbstractTaskDataset):
    name = "superglue-copa"
    labels_list = ['0', '1']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    def load_dataset(self, split):
        temp = load_from_disk("/root/datasets/copa")
        
        return temp[split]

    def preprocessor(self, example, add_prefix=True):
        
        src_texts = ["premise:", example["premise"],
                    "choice1:", example["choice1"],
                    "choice2:", example["choice2"]]
        # src_text = example['premise'] + "This happened because...\n\n Help me pick the more plausible option:\n\n-" + example['choice1'] + "\n-" + example['choice2']
        # tgt_text = [example['choice1'] if str(example["label"])=="1" else example['choice2']]
        tgt_text = [str(example["label"])]
        #return self.seq2seq_format(src_texts, tgt_texts, add_prefix)
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)
    
# class SuperGLUERecord(AbstractTaskDataset):
#     """Convert ReCoRD examples to text2text examples.
#     ReCoRD contains a passage, query containing a '@placeholder' string, and a set
#     of entities that are the possible values of the placeholder. Each train and
#     validation example will have a list of answers, any of which would be
#     considered correct.
#     For example, a typical example from ReCoRD might look like
#     {
#       'passsage': 'This is the passage.',
#       'query': 'A @placeholder is a bird.',
#       'entities': ['penguin', 'potato', 'pigeon'],
#       'answers': ['penguin', 'pigeon'],
#     }
#     which this preprocessor would turn into the following two examples:
#     {
#       'inputs': 'record query: A @placeholder is a bird. entities: penguin, '
#                 'potato, pigeon passage: This is the passage.',
#       'targets': 'penguin',
#     }
#     and
#     {
#       'inputs': 'record query: A @placeholder is a bird. entities: penguin, '
#                 'potato, pigeon passage: This is the passage.',
#       'targets': 'pigeon',
#     }
#     """
#     name = "superglue-record"
#     split_to_data_split = {"train": "train",
#                            "validation": "validation",
#                            "test": "validation"}
#     metric = [metrics.squad]
#     metric_names = ["squad"]

#     def load_dataset(self, split):
#         temp = load_from_disk("/root/datasets/record")
        
#         return temp[split]

#     def preprocessor(self, batch, add_prefix=True):
#         new_batch = collections.defaultdict(list)
#         keys = batch.keys()
#         for values in zip(*batch.values()):
#             ex = {k: v for k, v in zip(keys, values)}
#             # updates the passage.
#             passage = ex['passage']
#             passage = re.sub(
#                 r'(\.|\?|\!|\"|\')\n@highlight\n', r'\1 ', passage)
#             passage = re.sub(r'\n@highlight\n', '. ', passage)
#             inputs = f"record query: {ex['query']} entities: {', '.join(ex['entities'])} passage: {passage}"
#             if add_prefix:
#                 inputs = self.name + " " + inputs
#             # duplicates the samples based on  number of answers.
#             num_answers = len(ex["answers"])
#             num_duplicates = np.maximum(1, num_answers)
#             new_batch["source"].extend([inputs] * num_duplicates)
#             new_batch["target"].extend(
#                 ex["answers"] if num_answers > 0 else ["<unk>"])
#             new_batch["task"].extend([self.name] * num_duplicates)
#             new_batch["extra_fields"].extend(
#                 [{"answers": ex["answers"]}]*num_duplicates)
#         return new_batch

#     def map_dataset(self, dataset, add_prefix=True):
#         return dataset.map(functools.partial(self.preprocessor, add_prefix=add_prefix),
#                            batched=True, remove_columns=dataset.column_names)
   

    
class SuperGLUEBoolQTaskDataset(AbstractTaskDataset):
    name = "superglue-boolq"
    label_list = ["0", "1"]
    task_specific_config = {"max_length": compute_task_max_decoding_length(label_list)}
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }
    metrics = [metrics.accuracy]

    def load_dataset(self, split):
        temp = load_from_disk("/root/datasets/boolq")
        
        return temp[split]

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["question:", example["question"], "passage:", example["passage"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SuperGLUERTETaskDataset(AbstractTaskDataset):
    name = "superglue-rte"
    label_list = ["0", "1"]
    task_specific_config = {"max_length": compute_task_max_decoding_length(label_list)}
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }
    metrics = [metrics.accuracy]

    def load_dataset(self, split):
        return datasets.load_dataset(
            "super_glue", "rte", split=split
        )

    def preprocessor(self, example, add_prefix=True):
        src_texts = [
            "premise:",
            example["premise"],
            "hypothesis:",
            example["hypothesis"],
        ]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SuperGLUECBTaskDataset(AbstractTaskDataset):
    name = "superglue-cb"
    label_list = ["0", "1", "2"]
    task_specific_config = {"max_length": compute_task_max_decoding_length(label_list)}
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }
    metrics = [metrics.accuracy]

    def load_dataset(self, split):
        temp = load_from_disk("/root/datasets/cb")
        
        return temp[split]

    def preprocessor(self, example, add_prefix=True):
        src_texts = [
            "premise:",
            example["premise"],
            "hypothesis:",
            example["hypothesis"],
        ]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SNLITaskDataset(AbstractTaskDataset):
    name = "snli"
    label_list = ["0", "1", "2"]
    task_specific_config = {"max_length": compute_task_max_decoding_length(label_list)}
    metrics = [metrics.accuracy]

    def preprocessor(self, example, add_prefix=True):
        src_texts = [
            "premise:",
            example["premise"],
            "hypothesis:",
            example["hypothesis"],
        ]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class IWSLT2017RONL(AbstractTaskDataset):
    name = "iwslt2017-ro-nl"
    task_specific_config = {"max_length": 300, "num_beams": 4}
    pair = f"ro-nl"
    metrics = [metrics.bleu]

    def load_dataset(self, split):
        return datasets.load_dataset(
            "iwslt2017", "iwslt2017-ro-nl", split=split
        )

    def preprocessor(self, example, add_prefix=True):
        src_texts = [example["translation"]["ro"]]
        tgt_texts = [example["translation"]["nl"]]
        return self.seq2seq_format(
            src_texts, tgt_texts, add_prefix, prefix="Translate Romanian to Dutch"
        )


class IWSLT2017ENNL(AbstractTaskDataset):
    name = "iwslt2017-en-nl"
    task_specific_config = {"max_length": 300, "num_beams": 4}
    pair = f"en-nl"
    metrics = [metrics.bleu]

    def load_dataset(self, split):
        return datasets.load_dataset(
            "iwslt2017", "iwslt2017-en-nl", split=split
        )

    def preprocessor(self, example, add_prefix=True):
        src_texts = [example["translation"]["en"]]
        tgt_texts = [example["translation"]["nl"]]
        return self.seq2seq_format(
            src_texts, tgt_texts, add_prefix, prefix="Translate English to Dutch"
        )


class WMT16ENROTaskDataset(AbstractTaskDataset):
    name = "wmt16-en-ro"
    task_specific_config = {"max_length": 300, "num_beams": 4}
    pair = f"ro-en"
    metrics = [metrics.bleu]

    def load_dataset(self, split):
        return datasets.load_dataset(
            "wmt16", self.pair, split=split
        )

    def preprocessor(self, example, add_prefix=True):
        src_texts = [example["translation"]["en"]]
        tgt_texts = [example["translation"]["ro"]]
        return self.seq2seq_format(
            src_texts, tgt_texts, add_prefix, prefix="Translate English to Romanian"
        )


class WMT16ROENTaskDataset(AbstractTaskDataset):
    name = "wmt16-ro-en"
    task_specific_config = {"max_length": 300, "num_beams": 4}
    pair = f"ro-en"
    metrics = [metrics.bleu]

    def load_dataset(self, split):
        return datasets.load_dataset(
            "wmt16", self.pair, split=split
        )

    def preprocessor(self, example, add_prefix=True):
        src_texts = [example["translation"]["ro"]]
        tgt_texts = [example["translation"]["en"]]
        return self.seq2seq_format(
            src_texts, tgt_texts, add_prefix, prefix="Translate Romanian to English"
        )


class WMT16ENCSTaskDataset(AbstractTaskDataset):
    name = "wmt16-en-cs"
    task_specific_config = {"max_length": 300, "num_beams": 4}
    pair = f"cs-en"
    metrics = [metrics.bleu]

    def load_dataset(self, split):
        return datasets.load_dataset(
            "wmt16", self.pair, split=split
        )

    def preprocessor(self, example, add_prefix=True):
        src_texts = [example["translation"]["en"]]
        tgt_texts = [example["translation"]["cs"]]
        return self.seq2seq_format(
            src_texts, tgt_texts, add_prefix, prefix="Translate English to Czech"
        )


class WMT16ENFITaskDataset(AbstractTaskDataset):
    name = "wmt16-en-fi"
    task_specific_config = {"max_length": 300, "num_beams": 4}
    pair = f"fi-en"
    metrics = [metrics.bleu]

    def load_dataset(self, split):
        return datasets.load_dataset(
            "wmt16", self.pair, split=split
        )

    def preprocessor(self, example, add_prefix=True):
        src_texts = [example["translation"]["en"]]
        tgt_texts = [example["translation"]["fi"]]
        return self.seq2seq_format(
            src_texts, tgt_texts, add_prefix, prefix="Translate English to Finnish"
        )


class WMT14HIENTaskDataset(AbstractTaskDataset):
    name = "wmt14-hi-en"
    task_specific_config = {"max_length": 300, "num_beams": 4}
    pair = f"hi-en"
    metrics = [metrics.bleu]

    def load_dataset(self, split):
        return datasets.load_dataset(
            "wmt14", self.pair, split=split
        )

    def preprocessor(self, example, add_prefix=True):
        src_texts = [example["translation"]["en"]]
        tgt_texts = [example["translation"]["hi"]]
        return self.seq2seq_format(
            src_texts, tgt_texts, add_prefix, prefix="Translate English to Hindi"
        )


class TRECTaskDataset(AbstractTaskDataset):
    name = "trec"
    label_list = ["0", "1", "2", "3", "4", "5"]
    task_specific_config = {"max_length": compute_task_max_decoding_length(label_list)}
    split_to_data_split = {"train": "train", "validation": "test", "test": "test"}
    metrics = [metrics.accuracy]

    def load_dataset(self, split):
        return datasets.load_dataset("trec", split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence:", example["text"]]
        tgt_texts = [str(example["label-coarse"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class YelpPolarityTaskDataset(AbstractTaskDataset):
    name = "yelp_polarity"
    label_list = ["0", "1"]
    task_specific_config = {"max_length": compute_task_max_decoding_length(label_list)}
    split_to_data_split = {"train": "train", "validation": "test", "test": "test"}
    metrics = [metrics.accuracy]

    def load_dataset(self, split):
        return datasets.load_dataset(
            "yelp_polarity", split=split
        )

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence:", example["text"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class ScitailTaskDataset(AbstractTaskDataset):
    name = "scitail"
    label_list = ["0", "1"]
    task_specific_config = {"max_length": compute_task_max_decoding_length(label_list)}
    metrics = [metrics.accuracy]
    label_map = {"entailment": 0, "neutral": 1}

    def map_label(self, label):
        return self.label_map[label]

    def load_dataset(self, split):
        return datasets.load_dataset(
            "scitail", "snli_format", split=split
        )

    def preprocessor(self, example, add_prefix=True):
        src_texts = [
            "sentence1:",
            example["sentence1"],
            "sentence2:",
            example["sentence2"],
        ]
        # To increase the transfer performance, we modified the targets to be similar to other datasets.
        tgt_texts = [str(self.map_label(example["gold_label"]))]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class MRPCTaskDataset(AbstractTaskDataset):
    name = "mrpc"
    label_list = ["0", "1"]
    task_specific_config = {"max_length": compute_task_max_decoding_length(label_list)}
    metrics = [metrics.f1_score_with_invalid, metrics.accuracy]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }

    def load_dataset(self, split):
        # return datasets.load_dataset(
        #     "glue", "mrpc", split=split
        # )
        temp = load_from_disk("/root/datasets/mrpc")
        
        return temp[split]
    def preprocessor(self, example, add_prefix=True):
        src_texts = [
            "sentence1:",
            example["sentence1"],
            "sentence2:",
            example["sentence2"],
        ]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class COLATaskDataset(AbstractTaskDataset):
    name = "cola"
    label_list = ["0", "1"]
    task_specific_config = {"max_length": compute_task_max_decoding_length(label_list)}
    metrics = [metrics.matthews_corrcoef]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }

    def load_dataset(self, split):
        # return datasets.load_dataset(
        #     "glue", "cola", split=split
        # )
        temp = load_from_disk("/root/datasets/cola")
        
        return temp[split]
    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence:", example["sentence"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SST2TaskDataset(AbstractTaskDataset):
    name = "sst2"
    label_list = ["0", "1"]
    task_specific_config = {"max_length": compute_task_max_decoding_length(label_list)}
    metrics = [metrics.accuracy]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }

    def load_dataset(self, split):
        # return datasets.load_dataset(
        #     "glue", "sst2", split=split
        # )
        temp = load_from_disk("/root/datasets/sst2")
        
        return temp[split]
    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence:", example["sentence"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class STSBTaskDataset(AbstractTaskDataset):
    name = "stsb"
    label_list = [str(np.round(label, decimals=1)) for label in np.arange(0, 5.2, 0.2)]
    task_specific_config = {"max_length": compute_task_max_decoding_length(label_list)}
    metrics = [metrics.pearson_corrcoef, metrics.spearman_corrcoef]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }

    def load_dataset(self, split):
        # return datasets.load_dataset(
        #     "glue", "stsb", split=split
        # )
        temp = load_from_disk("/root/datasets/stsb")
        
        return temp[split]        
    def preprocessor(self, example, add_prefix=True):
        src_texts = [
            "sentence1:",
            example["sentence1"],
            "sentence2:",
            example["sentence2"],
        ]
        tgt_texts = [str(round_stsb_target(example["label"]))]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class QQPTaskDataset(AbstractTaskDataset):
    name = "qqp"
    label_list = ["0", "1"]
    task_specific_config = {"max_length": compute_task_max_decoding_length(label_list)}
    metrics = [metrics.f1_score_with_invalid, metrics.accuracy]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }

    def load_dataset(self, split):
        # return datasets.load_dataset(
        #     "glue", "qqp", split=split
        # )
        temp = load_from_disk("/root/datasets/qqp")
        
        return temp[split]
    def preprocessor(self, example, add_prefix=True):
        src_texts = [
            "question1:",
            example["question1"],
            "question2:",
            example["question2"],
        ]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class MNLITaskDataset(AbstractTaskDataset):
    name = "mnli"
    label_list = ["0", "1", "2"]
    task_specific_config = {"max_length": compute_task_max_decoding_length(label_list)}
    split_to_data_split = {
        "train": "train",
        "validation": "validation_mismatched",
        "test": "validation_matched",
    }
    metrics = [metrics.accuracy]

    def load_dataset(self, split):
        # return datasets.load_dataset(
        #     "glue", "mnli", split=split
        # )
        temp = load_from_disk("/root/datasets/mnli")
        
        return temp[split]
    def preprocessor(self, example, add_prefix=True):
        src_texts = [
            "premise:",
            example["premise"],
            "hypothesis:",
            example["hypothesis"],
        ]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class AdversarialNLITaskDataset(AbstractTaskDataset):
    name = "anli"
    label_list = ["0", "1", "2"]
    task_specific_config = {"max_length": compute_task_max_decoding_length(label_list)}
    split_to_data_split = {
        "train": "train",
        "validation": "dev",
        "test": "test",
    }
    metrics = [metrics.accuracy]
    suffixes = ["_r1", "_r2", "_r3"]

    def load_dataset(self, split):
        # anli has 3 subsplits, we just combine these following exT5
        subsplits = []
        for suffix in self.suffixes:
            subsplits.append(datasets.load_dataset(self.name, split=split + suffix))
        return datasets.concatenate_datasets(subsplits)

    def preprocessor(self, example, add_prefix=True):
        src_texts = [
            "premise:",
            example["premise"],
            "hypothesis:",
            example["hypothesis"],
        ]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class AbductiveNLITaskDataset(AbstractTaskDataset):
    name = "art"
    label_list = ["0", "1", "2"]
    task_specific_config = {"max_length": compute_task_max_decoding_length(label_list)}
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }
    metrics = [metrics.accuracy]

    def preprocessor(self, example, add_prefix=True):
        src_texts = [
            "observation 1:",
            example["observation_1"],
            "observation 2:",
            example["observation_2"],
            "hypothesis 1:",
            example["hypothesis_1"],
            "hypothesis 2:",
            example["hypothesis_2"],
        ]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class QNLITaskDataset(AbstractTaskDataset):
    name = "qnli"
    label_list = ["0", "1"]
    task_specific_config = {"max_length": compute_task_max_decoding_length(label_list)}
    metrics = [metrics.accuracy]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }

    def load_dataset(self, split):
        # return datasets.load_dataset(
        #     "glue", "qnli", split=split
        # )
        temp = load_from_disk("/root/datasets/qnli")
        
        return temp[split]
    def preprocessor(self, example, add_prefix=True):
        src_texts = ["question:", example["question"], "sentence:", example["sentence"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class RTETaskDataset(AbstractTaskDataset):
    name = "rte"
    label_list = ["0", "1"]
    task_specific_config = {"max_length": compute_task_max_decoding_length(label_list)}
    metrics = [metrics.accuracy]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }

    def load_dataset(self, split):
        # return datasets.load_dataset(
        #     "glue", "rte", split=split
        # )
        temp = load_from_disk("/root/datasets/rte")
        
        return temp[split]
    def preprocessor(self, example, add_prefix=True):
        src_texts = [
            "sentence1:",
            example["sentence1"],
            "sentence2:",
            example["sentence2"],
        ]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class WNLITaskDataset(AbstractTaskDataset):
    name = "wnli"
    label_list = ["0", "1"]
    task_specific_config = {"max_length": compute_task_max_decoding_length(label_list)}
    metrics = [metrics.accuracy]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }

    def load_dataset(self, split):
        # return datasets.load_dataset(
        #     "glue", "wnli", split=split
        # )
        temp = load_from_disk("/root/datasets/wnli")
        
        return temp[split]
    def preprocessor(self, example, add_prefix=True):
        src_texts = [
            "sentence1:",
            example["sentence1"],
            "sentence2:",
            example["sentence2"],
        ]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SocialIQaTaskDataset(AbstractTaskDataset):
    name = "social_i_qa"
    label_map = {"1": "0", "2": "1", "3": "2"}
    label_list = ["0", "1", "2"]
    task_specific_config = {"max_length": compute_task_max_decoding_length(label_list)}
    metrics = [metrics.accuracy]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }

    def preprocessor(self, example, add_prefix=True):
        src_texts = [
            "question:",
            example["question"],
            "context:",
            example["context"],
            "answerA:",
            example["answerA"],
            "answerB:",
            example["answerB"],
            "answerC:",
            example["answerC"],
        ]
        tgt_texts = [example["label"].rstrip()]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class CosmosQaTaskDataset(AbstractTaskDataset):
    name = "cosmos_qa"
    label_list = ["0", "1", "2", "3"]
    task_specific_config = {"max_length": compute_task_max_decoding_length(label_list)}
    metrics = [metrics.accuracy]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }

    def preprocessor(self, example, add_prefix=True):
        src_texts = [
            "question:",
            example["question"],
            "context:",
            example["context"],
            "answer0:",
            example["answer0"],
            "answer1:",
            example["answer1"],
            "answer2:",
            example["answer2"],
            "answer3:",
            example["answer3"],
        ]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class WinograndeTaskDataset(AbstractTaskDataset):
    name = "winogrande"
    label_list = ["1", "2"]
    task_specific_config = {"max_length": compute_task_max_decoding_length(label_list)}
    metrics = [metrics.accuracy]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }

    def load_dataset(self, split):
        return datasets.load_dataset(
            "winogrande", "winogrande_l", split=split
        )

    def preprocessor(self, example, add_prefix=True):
        src_texts = [
            "sentence:",
            example["sentence"],
            "option1:",
            example["option1"],
            "option2:",
            example["option2"],
        ]
        tgt_texts = [str(example["answer"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class HellaSwagTaskDataset(AbstractTaskDataset):
    name = "hellaswag"
    label_list = ["0", "1", "2", "3"]
    task_specific_config = {"max_length": compute_task_max_decoding_length(label_list)}
    metrics = [metrics.accuracy]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }

    def preprocessor(self, example, add_prefix=True):
        src_texts = [
            "ctx:",
            example["ctx"],
            "ending0:",
            example["endings"][0],
            "ending1:",
            example["endings"][1],
            "ending2:",
            example["endings"][2],
            "ending3:",
            example["endings"][3],
        ]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class CommonsenseQaTaskDataset(AbstractTaskDataset):
    name = "commonsense_qa"
    label_list = ["A", "B", "C", "D", "E"]
    task_specific_config = {"max_length": compute_task_max_decoding_length(label_list)}
    metrics = [metrics.accuracy]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }

    def preprocessor(self, example, add_prefix=True):
        src_texts = [
            "question:",
            example["question"],
            "A:",
            example["choices"]["text"][0],
            "B:",
            example["choices"]["text"][1],
            "C:",
            example["choices"]["text"][2],
            "D:",
            example["choices"]["text"][3],
            "E:",
            example["choices"]["text"][4],
        ]
        tgt_texts = [example["answerKey"]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SquadDataset(AbstractTaskDataset):
    name = "squad"
    task_specific_config = {
        "max_length": 16
    }  # based on 't5 on tpu' notebook, nothing special.
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "validation",
    }
    metrics = [metrics.squad_metrics]
    generation_task = True

    def preprocessor(self, example, add_prefix=False):
        src_texts = ["question:", example["question"], "context:", example["context"]]
        tgt_texts = [str(example["answers"]["text"][0])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix, id=example["id"])


class MrqaDataset(AbstractTaskDataset):
    name = "mrqa"
    task_specific_config = {"max_length": 64}
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "test",
    }
    metrics = [metrics.squad_metrics]
    generation_task = True

    def preprocessor(self, example, add_prefix=False):
        src_texts = ["question:", example["question"], "context:", example["context"]]
        tgt_texts = [str(example["answers"][0])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix, id=example["qid"])


# # mrqa with chunking preprocessing. Kept separate in case we want to use unchunked.
# class ChunkedMrqaDataset(AbstractTaskDataset):
#     name = "mrqa"
#     task_specific_config = {"max_length": 64}
#     split_to_data_split = {
#         "train": "train",
#         "validation": "validation",
#         "test": "test",
#     }
#     metrics = [metrics.squad_metrics]
#     generation_task = True

#     def __init__(self, seed=42, tokenizer=T5TokenizerFast.from_pretrained("t5-base")):
#         self.seed = seed
#         self.tokenizer = tokenizer
#         self.subsets = [
#             "HotpotQA",
#             "NaturalQuestionsShort",
#             "NewsQA",
#             "SearchQA",
#             "SQuAD",
#             "TriviaQA-web",
#         ]
#         self.filter_nulls = False

#     def toggle_null_filter(self):
#         self.filter_nulls = not self.filter_nulls

#     # TODO: this is fairly hideous, i pull out of batched form, add my rows,
#     # then put it back in. There is probably a more efficient way to do this...
#     def preprocessor(self, samples, split, add_prefix=False):
#         examples = []
#         result = defaultdict(list)
#         for i in range(len(samples["qid"])):
#             examples.append({k: samples[k][i] for k in samples})
#         for sample in examples:
#             for chunked_sample in chunk_sample(
#                 self.tokenizer,
#                 sample,
#                 filter_nulls=self.filter_nulls and split == "train",
#             ):
#                 for key in chunked_sample:
#                     result[key].append(chunked_sample[key])
#         # little bit of housekeeping
#         result["id"] = result["qid"]
#         result["task"] = result["subset"]
#         return result

#     def get_dataset(
#         self, split, n_obs=None, add_prefix=False, split_validation_test=False
#     ):
#         dataset = self.get_shuffled_sampled_split(split, n_obs)
#         # downsample similar to MADE & MRQA baselines.
#         # if split == 'train' or split == 'validation':
#         #     all_datasubsets = []
#         #     # we downsample to 75k train, 1k val examples. No test downsample.
#         #     downsample_size = 75000 if split == "train" else 1000
#         #     for subset in self.subsets:
#         #         datasubset = dataset.filter(lambda x: x["subset"] == subset)
#         #         if len(datasubset) > downsample_size:
#         #             datasubset = datasubset.shuffle().select(range(downsample_size))
#         #         all_datasubsets.append(datasubset)
#         #     # we want to sample from each dataset uniformly
#         #     probabilities = [1 / len(all_datasubsets) for _ in all_datasubsets]
#         #     dataset = datasets.interleave_datasets(all_datasubsets, probabilities=probabilities)
#         # apply chunking and prepro
#         dataset = dataset.map(
#             functools.partial(self.preprocessor, split=split, add_prefix=add_prefix),
#             remove_columns=dataset.column_names,
#             batched=True,  # so we can add rows.
#         )
#         return dataset


class XSumTaskDataset(AbstractTaskDataset):
    name = "xsum"
    task_specific_config = {"max_length": 60, "min_length": 10, "num_beams": 6}
    metrics = [metrics.rouge]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "test",
    }
    generation_task = True

    def preprocessor(self, example, add_prefix=True):
        src_texts = [example["document"]]
        tgt_texts = [str(example["summary"])]
        return self.seq2seq_format(
            src_texts, tgt_texts, add_prefix, prefix="summarize:"
        )


class CnnDailyMailDataset(AbstractTaskDataset):
    name = "cnn_dailymail"
    task_specific_config = {"max_length": 60, "min_length": 10, "num_beams": 4}
    metrics = [metrics.rouge]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "test",
    }
    generation_task = True

    def load_dataset(self, split: int):
        return datasets.load_dataset(
            'ccdv/cnn_dailymail', '3.0.0', split=split
        )

    def preprocessor(self, example, add_prefix=True):
        src_texts = [example["article"]]
        tgt_texts = [example["highlights"]]
        return self.seq2seq_format(
            src_texts, tgt_texts, add_prefix, prefix="summarize:"
        )


class WikiLinguaDataset(AbstractTaskDataset):
    name = "wiki_lingua_english_en"
    task_specific_config = {"max_length": 60, "min_length": 10, "num_beams": 4}
    metrics = [metrics.rouge]
    split_to_data_split = {
        "train": "train",
        "validation": "validation",
        "test": "test",
    }
    generation_task = True

    def load_dataset(self, split: int):
        return datasets.load_dataset(
            "gem", self.name, split=split
        )

    def preprocessor(self, example, add_prefix=True):
        src_texts = [example["source"]]
        tgt_texts = [str(example["target"])]
        return self.seq2seq_format(
            src_texts, tgt_texts, add_prefix, prefix="summarize:"
        )


    


TASK_MAPPING = OrderedDict(
    [
        ("superglue-boolq", SuperGLUEBoolQTaskDataset),
        ("superglue-cb", SuperGLUECBTaskDataset),
        ("superglue-rte", SuperGLUERTETaskDataset),
        ('superglue-multirc', SuperGLUEMultiRC),
        ('superglue-wic', SuperGLUEWIC),
        ('superglue-wsc.fixed', SuperGLUEWSCFixed),
        # ('superglue-record', SuperGLUERecord),
        ('superglue-copa', SuperGLUECOPA),
        ("paws", PawsTaskDataset),
        ("imdb", IMDBTaskDataset),
        ("snli", SNLITaskDataset),
        ("scitail", ScitailTaskDataset),
        ("mrpc", MRPCTaskDataset),
        ("trec", TRECTaskDataset),
        ("yelp_polarity", YelpPolarityTaskDataset),
        ("wmt16-ro-en", WMT16ROENTaskDataset),
        ("wmt14-hi-en", WMT14HIENTaskDataset),
        ("wmt16-en-ro", WMT16ENROTaskDataset),
        ("wmt16-ro-en", WMT16ROENTaskDataset),
        ("wmt16-en-cs", WMT16ENCSTaskDataset),
        ("iwslt2017-ro-nl", IWSLT2017RONL),
        ("iwslt2017-en-nl", IWSLT2017ENNL),
        ("cola", COLATaskDataset),
        ("sst2", SST2TaskDataset),
        ("stsb", STSBTaskDataset),
        ("qqp", QQPTaskDataset),
        ("mnli", MNLITaskDataset),
        ("qnli", QNLITaskDataset),
        ("rte", RTETaskDataset),
        ("wnli", WNLITaskDataset),
        ("wmt16-en-fi", WMT16ENFITaskDataset),
        ("social_i_qa", SocialIQaTaskDataset),
        ("cosmos_qa", CosmosQaTaskDataset),
        ("winogrande", WinograndeTaskDataset),
        ("hellaswag", HellaSwagTaskDataset),
        ("commonsense_qa", CommonsenseQaTaskDataset),
        ("sick", SickTaskDataset),
        ("squad", SquadDataset),
        ("mrqa_reg", MrqaDataset),
        ("xsum", XSumTaskDataset),
        ("cnn_dailymail", CnnDailyMailDataset),
        ("wiki_lingua_english_en", WikiLinguaDataset),
        ("anli", AdversarialNLITaskDataset),
        ("art", AbductiveNLITaskDataset),
    ]
)


class AutoTask:
    @classmethod
    def get(self, task_name, seed=42):
        if task_name in TASK_MAPPING:
            return TASK_MAPPING[task_name](seed)
        raise ValueError(
            "Unrecognized task {} for AutoTask Model.\n"
            "Task name should be one of {}.".format(
                task_name, ", ".join(c for c in TASK_MAPPING.keys())
            )
        )
