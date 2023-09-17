import os
import sys
import math
import copy
from transformers import AutoTokenizer
from collections import defaultdict


class UnigramTokenizer():
    def __init__(self, pre_tokenizer, model=None):
        self.pre_tokenizer = pre_tokenizer
        self.model = model
    
    def tokenize(self, text):
        words = [word for word, _ in self.pre_tokenizer(text)]
        encoded_words = [self._encode_word(word, self.model)[0] for word in words]
        return sum(encoded_words, [])

    def train(self, corpus, vocab_size):
        words_list = [self.pre_tokenizer(text) for text in corpus]
        word2count = defaultdict(int)
        for words in words_list:
            for w, _ in words:
                word2count[w] += 1
        
        char2count = defaultdict(int)
        subword2count = defaultdict(int)
        for word, count in word2count.items():
            for i in range(len(word)):
                char2count[word[i]] += count
                for j in range(i + 2, len(word) + 1):
                    subword2count[word[i: j]] += count
        sorted_subwords = sorted(subword2count.items(), key=lambda x: x[1], reverse=True)
        tokens = list(char2count.items()) + sorted_subwords[:300 - len(char2count)]
        print(tokens)

        token2count = {token: count for token, count in tokens}
        total_count = sum([count for _, count in token2count.items()])
        model = {token: -math.log(count / total_count) for token, count in token2count.items()}
        print(model)
        percent_to_remove = 0.1

        while len(model) > vocab_size:
            scores = self._compute_scores(model, word2count)
            sorted_scores = sorted(scores.items(), key=lambda x: x[1])
            # Remove percent_to_remove tokens with the lowest scores.
            for i in range(int(len(model) * percent_to_remove)):
                _ = token2count.pop(sorted_scores[i][0])
            total_count = sum([count for _, count in token2count.items()])
            model = {token: -math.log(count / total_count) for token, count in token2count.items()}
        
        self.model = model
    
    def _compute_scores(self, model, word2count):
        scores = {}
        model_loss = self._compute_loss(model, word2count)
        for token, score in model.items():
            if len(token) == 1:
                continue
            model_without_token = copy.deepcopy(model)
            _ = model_without_token.pop(token)
            scores[token] = self._compute_loss(model_without_token, word2count) - model_loss
        
        return scores

    def _compute_loss(self, model, word2count):
        loss = 0
        for word, count in word2count.items():
            _, word_loss = self._encode_word(word, model)
            loss += word_loss
        
        return loss

    def _encode_word(self, word, model):
        best_segmentations = [{"start": 0, "score": 1}] + [{"start": None, "score": None} for _ in range(len(word))]
        for start_idx in range(len(word)):
            # This should be properly filled by the previous steps of the loop
            best_score_at_start = best_segmentations[start_idx]["score"]
            for end_idx in range(start_idx + 1, len(word) + 1):
                token = word[start_idx:end_idx]
                if token in model and best_score_at_start is not None:
                    score = model[token] + best_score_at_start
                    # If we have found a better segmentation (lower score) ending at end_idx
                    if (
                            best_segmentations[end_idx]["score"] is None
                            or best_segmentations[end_idx]["score"] > score
                    ):
                        best_segmentations[end_idx] = {"start": start_idx, "score": score}
        segmentation = best_segmentations[-1]
        if segmentation["score"] is None:
            # We did not find a tokenization of the word -> unknown
            return ["<unk>"], None
        score = segmentation["score"]
        start = segmentation["start"]
        end = len(word)
        tokens = []
        while start != 0:
            tokens.insert(0, word[start:end])
            next_start = best_segmentations[start]["start"]
            end = start
            start = next_start
        tokens.insert(0, word[start:end])
        return tokens, score



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
    vocab_size = 100

    tokenizer = AutoTokenizer.from_pretrained("../../pretrained_model/t5-large")
    pre_tokenize_function = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str
    tokenizer = UnigramTokenizer(pre_tokenize_function)
    tokenizer.train(corpus, vocab_size)

    tokens = tokenizer.tokenize("The cat sat on the mat.")
    print(tokens)


if __name__ == "__main__":
    test()