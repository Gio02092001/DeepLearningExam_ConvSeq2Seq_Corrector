import pickle
import string
import random

import chardet
import nltk
import pandas as pd
from nltk import word_tokenize, RegexpTokenizer, punkt
from tqdm import tqdm


#nltk.download('punkt')

class BuildDictionary_Map:
    tokenizer = RegexpTokenizer(r'[^.!?]+[.!?]')
    """ Manca la generazione di 0.1x10x500.000 e 0.1x10x3.500.000"""
    corruption_prob = 0.1
    times = 5
    sentenceNumber = 100000

    def __init__(self, sentence, rep, p):
        self.sourceSOS = self.sourceEOS = self.sourcePAD = self.sourceUNK = 0
        self.targetSOS = self.targetEOS = self.targetPAD = 0
        self.corruption_prob=p
        self.sentenceNumber = sentence
        self.times=rep

    def loadDictionaries(self, sentence, rep, p):
        """
        Load precomputed dictionaries.
        """

        def load_pickle(filename, special_tokens,index_to_Target):
            try:
                with open(filename, 'rb') as f:
                    dictionary = pickle.load(f)
                    for token in special_tokens:
                        if index_to_Target:
                            dictionary[len(dictionary) + 1]=token
                        else:
                            dictionary[token] = len(dictionary) + 1
                    return dictionary
            except FileNotFoundError:
                tqdm.write(f"The file '{filename}' was not found.")
                return {}

        index_to_word_dict = load_pickle(f'data/dictionaries/{sentence}_index_to_word.pkl', ["<sos>", "<eos>", "<pad>", "<unk>"],
                                True)
        word_dict = load_pickle(f'data/dictionaries/{sentence}_word_to_index.pkl', ["<sos>", "<eos>", "<pad>", "<unk>"], False)
        target_word_dict = load_pickle(f'data/dictionaries/{sentence}_target_word_to_index.pkl',
                                       ["<sos>", "<eos>", "<pad>"], False)
        index_to_target_word_dict=load_pickle(f'data/dictionaries/{sentence}_index_to_target_word.pkl',
                                              ["<sos>", "<eos>", "<pad>"],True)

        self.sourceSOS = word_dict["<sos>"]
        self.sourceEOS = word_dict["<eos>"]
        self.sourcePAD = word_dict["<pad>"]
        self.sourceUNK = word_dict["<unk>"]
        self.targetSOS = target_word_dict["<sos>"]
        self.targetEOS = target_word_dict["<eos>"]
        self.targetPAD = target_word_dict["<pad>"]

        try:
            with open(f'data/dictionaries/{sentence}x{rep}x{p}_SentenceMap.pkl', 'rb') as f:
                sentenceMap = pickle.load(f)
            tqdm.write("Sentence map has been loaded.")
        except FileNotFoundError:
            tqdm.write(f"The file '{sentence}x{rep}x{p}SentenceMap.pkl' was not found.")
            sentenceMap = {}

        return word_dict, target_word_dict, sentenceMap, index_to_target_word_dict, index_to_word_dict

    def corrupt_word_multiple(self, word, corruption_prob=None):
        """
        Randomly corrupts each character in a word with a given probability.
        """
        SPECIAL_TOKENS = {"<UNK>", "<unk>", "<eos>", "<sos>", "<pad>"}
        if word in SPECIAL_TOKENS:
            return word
        else:
            keyboard_neighbors = {
                'q': ['w', 'a'],
                'w': ['q', 'e', 'a', 's'],
                'e': ['w', 'r', 's', 'd'],
                'r': ['e', 't', 'd', 'f'],
                't': ['r', 'y', 'f', 'g'],
                'y': ['t', 'u', 'g', 'h'],
                'u': ['y', 'i', 'h', 'j'],
                'i': ['u', 'o', 'j', 'k'],
                'o': ['i', 'p', 'k', 'l'],
                'p': ['o', 'l'],

                'a': ['q', 'w', 's', 'z'],
                's': ['a', 'w', 'd', 'z', 'x'],
                'd': ['s', 'e', 'f', 'x', 'c'],
                'f': ['d', 'r', 'g', 'c', 'v'],
                'g': ['f', 't', 'h', 'v', 'b'],
                'h': ['g', 'y', 'j', 'b', 'n'],
                'j': ['h', 'u', 'k', 'n', 'm'],
                'k': ['j', 'i', 'l', 'm'],
                'l': ['k', 'o'],

                'z': ['a', 's', 'x'],
                'x': ['z', 's', 'd', 'c'],
                'c': ['x', 'd', 'f', 'v'],
                'v': ['c', 'f', 'g', 'b'],
                'b': ['v', 'g', 'h', 'n'],
                'n': ['b', 'h', 'j', 'm'],
                'm': ['n', 'j', 'k'],

                'Q': ['W', 'A'],
                'W': ['Q', 'E', 'A', 'S'],
                'E': ['W', 'R', 'S', 'D'],
                'R': ['E', 'T', 'D', 'F'],
                'T': ['R', 'Y', 'F', 'G'],
                'Y': ['T', 'U', 'G', 'H'],
                'U': ['Y', 'I', 'H', 'J'],
                'I': ['U', 'O', 'J', 'K'],
                'O': ['I', 'P', 'K', 'L'],
                'P': ['O', 'L'],

                'A': ['Q', 'W', 'S', 'Z'],
                'S': ['A', 'W', 'D', 'Z', 'X'],
                'D': ['S', 'E', 'F', 'X', 'C'],
                'F': ['D', 'R', 'G', 'C', 'V'],
                'G': ['F', 'T', 'H', 'V', 'B'],
                'H': ['G', 'Y', 'J', 'B', 'N'],
                'J': ['H', 'U', 'K', 'N', 'M'],
                'K': ['J', 'I', 'L', 'M'],
                'L': ['K', 'O'],

                'Z': ['A', 'S', 'X'],
                'X': ['Z', 'S', 'D', 'C'],
                'C': ['X', 'D', 'F', 'V'],
                'V': ['C', 'F', 'G', 'B'],
                'B': ['V', 'G', 'H', 'N'],
                'N': ['B', 'H', 'J', 'M'],
                'M': ['N', 'J', 'K']
            }

            corruption_prob = corruption_prob or self.corruption_prob
            corrupted_word = []
            for char in word:
                if char =="-":
                    print(word)
                if random.random() < corruption_prob:
                    if (len(word)>1):
                        corruption_type = random.choice(['add', 'delete', 'change'])
                    else:
                        corruption_type = random.choice(['add', 'change'])
                    if corruption_type == 'add':
                        corrupted_word.append(char)
                        corrupted_word.append(random.choice(keyboard_neighbors[char]))
                    elif corruption_type == 'delete':
                        continue
                    elif corruption_type == 'change':
                        corrupted_word.append(random.choice(keyboard_neighbors[char]))
                corrupted_word.append(char)

            if random.random() < corruption_prob:
                corrupted_word.append(random.choice(string.ascii_letters))

            if len(''.join(corrupted_word))<1:
                return random.choice(string.ascii_letters)
            else:
                return ''.join(corrupted_word)

    def corrupt_sentence(self, words, corruption_prob=None, times=None):
        """
        Corrupts a sentence by applying corruption logic to each word in the sentence.
        """
        corruption_prob = corruption_prob or self.corruption_prob
        times = times or self.times
        return [' '.join(self.corrupt_word_multiple(word, corruption_prob) for word in words) for _ in range(times)]

    def buildDictionary(self):
        """
        Reads a dataset, tokenizes sentences, and builds word dictionaries.
        """
        tqdm.write("Reading file...")
        with open("data/WikiArticlesCorrect", "r", encoding="utf-16") as f:
            article = f.read()
        #encoding_info = chardet.detect(article)
        #tqdm.write(encoding_info)
        tqdm.write("Tokenizing sentences...")
        sentences = self.tokenizer.tokenize(article)

        all_words = []
        all_target_words=[]
        all_sentences = {}

        for sentence in tqdm(sentences[:self.sentenceNumber], desc="Processing sentences"):
            #tqdm.write(f"Processing sentence {counter + 1}/{len(sentences)} ({(counter + 1) / len(sentences) * 100:.2f}%)")
            words = [word for word in word_tokenize(sentence) if word not in string.punctuation]
            finalSentence = ' '.join(words)
            all_words.extend(words)
            all_target_words.extend(words)

            for corrupted_sentence in self.corrupt_sentence(words):
                corrupted_words = [word for word in word_tokenize(corrupted_sentence) if word not in string.punctuation]
                all_words.extend(corrupted_words)
                all_sentences.setdefault(corrupted_sentence, finalSentence)

        index_to_word = {idx: word for idx, word in enumerate(pd.Series(all_words).drop_duplicates())}
        word_to_index = {word: idx for idx, word in enumerate(pd.Series(all_words).drop_duplicates())}
        target_word_to_index = {word: idx for idx, word in enumerate(pd.Series(all_target_words).drop_duplicates())}
        index_to_target_word = {idx: word for idx, word in enumerate(pd.Series(all_target_words).drop_duplicates())}

        with open(f'data/dictionaries/{self.sentenceNumber}_word_to_index.pkl', 'wb') as f:
            pickle.dump(word_to_index, f)

        with open(f'data/dictionaries/{self.sentenceNumber}_target_word_to_index.pkl', 'wb') as f:
            pickle.dump(target_word_to_index, f)

        with open(f'data/dictionaries/{self.sentenceNumber}_index_to_target_word.pkl', 'wb') as f:
            pickle.dump(index_to_target_word, f)

        with open(f'data/dictionaries/{self.sentenceNumber}_index_to_word.pkl', 'wb') as f:
            pickle.dump(index_to_word, f)

        with open(f'data/dictionaries/{self.sentenceNumber}x{self.times}x{self.corruption_prob}_SentenceMap.pkl',
                  'wb') as f:
            pickle.dump(all_sentences, f)

        tqdm.write("Word-to-index dictionary has been saved successfully.")

    def splitSet(self, sentenceMap, validationSet):
        """
        Splits a dataset into training and validation sets.
        """
        random.seed(42)
        val_size = int(validationSet * len(sentenceMap))
        val_keys = random.sample(list(sentenceMap.keys()), val_size)

        total = len(sentenceMap)
        train_data = {}
        validation_data = {}

        for k, v in tqdm(sentenceMap.items(), desc="Splitting train/validation"):
            if k in val_keys:
                validation_data[k] = v
            else:
                train_data[k] = v

            # tqdm.write progress every 0.1%
            """if i % max(1, total // 1000) == 0:
                percent = (i / total) * 100
                tqdm.write(f"Progress: {percent:.3f}%")
"""
        tqdm.write("Completed splitting data!")

        tqdm.write(f"Train size: {len(train_data)}, Validation size: {len(validation_data)}")
        return train_data, validation_data