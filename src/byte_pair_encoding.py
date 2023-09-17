import os
import sys
import json
from sentencepiece import SentencePieceProcessor
from typing import List
from collections import defaultdict
from transformers import AutoTokenizer
from tqdm import tqdm


class BPETokenizer():
    def __init__(self, pre_tokenizer, vocabs=None, merge_rules=None):
        self.pre_tokenizer = pre_tokenizer
        self.vocabs = vocabs
        self.merge_rules = merge_rules

    def tokenize(self, text: str) -> List[str]:
        word_span = self.pre_tokenizer(text)
        words = [x for x, _ in word_span]
        splits_list = [[x for x in word] for word in words]
        
        for merge_rule in self.merge_rules:
            merged_chars = "".join(merge_rule)
            for idx, splits in enumerate(splits_list):
                j = 0
                while j < len(splits) - 1:
                    if splits[j] == merge_rule[0] and splits[j + 1] == merge_rule[1]:
                        splits = splits[:j] + [merged_chars] + splits[j + 2:]
                    else:
                        j += 1
                splits_list[idx] = splits
        
        tokens = []
        for splits in splits_list:
            tokens += splits
        
        return tokens
    

    def train(self, corpus, vocab_size, save_dir=None):
        
        words_list = [self.pre_tokenizer(text) for text in corpus]
        word2count = defaultdict(int)
        for words in words_list:
            for w, _ in words:
                word2count[w] += 1

        vocab_set = set()
        for word in word2count:
            vocab_set.update(list(word))
        vocabs = list(vocab_set)

        word2splits = {word: [x for x in word] for word in word2count}

        merge_rules = []

        while len(vocabs) < vocab_size:
            pair2count = self._compute_pair2count(word2count, word2splits)
            if len(pair2count) == 0:
                break
            best_pair = self._get_max_count_pair(pair2count)
            vocabs.append(best_pair[0] + best_pair[1])
            merge_rules.append(best_pair)
            word2splits = self._merge_pair(best_pair, word2splits)

        self.vocabs = vocabs
        self.merge_rules = merge_rules

        if save_dir is not None:
            state = {"vocabs": vocabs, "merge_rules": merge_rules}
            save_path = os.path.join(save_dir, "tokenizer_state.json")
            with open(save_path, "w") as f:
                json.dump(state, f, indent=4)

    
    def _compute_pair2count(self, word2count, word2splits):
        pair2count = defaultdict(int)
        for word, count in word2count.items():
            splits = word2splits[word]
            if len(splits) <= 1:
                continue
            for i in range(len(splits) - 1):
                pair2count[(splits[i], splits[i + 1])] += count
        return pair2count
    
    def _get_max_count_pair(self, pair2count):
        best_pair = None
        max_count = 0
        for pair, count in pair2count.items():
            if max_count is None or max_count < count:
                max_count = count
                best_pair = pair
        return best_pair
    
    def _merge_pair(self, pair, word2splits):
        new_word2splits = {}
        span = pair[0] + pair[1]
        for word, splits in word2splits.items():
            if len(splits) == 1:
                new_word2splits[word] = splits
            i = 0
            while i < len(splits) - 1:
                if splits[i] == pair[0] and splits[i + 1] == pair[1]:
                    splits = splits[:i] + [span] + splits[i+2:]
                else:
                    i += 1
            new_word2splits[word] = splits
        
        return new_word2splits
    

def test():
    os.chdir(sys.path[0])
    corpus = [
        "The cat sat on the mat.",
        "The cat did not sit on the mat.",
        "And if both apply, they are essentially impossible.",
        "And if both apply, they are essentially possible.",
        "The water is too hot.",
        "The water is too cold.",
    ]
    vocab_size = 50

    tokenizer = AutoTokenizer.from_pretrained("../../pretrained_model/t5-large")
    pre_tokenize_function = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str
    tokenizer = BPETokenizer(pre_tokenize_function)
    tokenizer.train(corpus, vocab_size)

    tokens = tokenizer.tokenize("The cat sat on the mat.")
    print(tokens)


if __name__ == "__main__":
    test()
