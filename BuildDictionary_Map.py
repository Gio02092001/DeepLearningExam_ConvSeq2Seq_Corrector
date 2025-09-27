import os
import pickle
import string
import random
import re

import chardet
import nltk
import pandas as pd
from nltk import word_tokenize, RegexpTokenizer, punkt
from tensorflow import timestamp
from tokenizers.processors import TemplateProcessing
from tqdm import tqdm
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors, normalizers
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from mosestokenizer import *



#nltk.download('punkt')

class BuildDictionary_Map:
    tokenizer = MosesTokenizer('en')
    """ Manca la generazione di 0.1x10x500.000 e 0.1x10x3.500.000"""
    corruption_prob = 0.1
    times = 5
    sentenceNumber = 100000

    def __init__(self, sentence, rep, p, bpe, timestamp):
        self.sourceSOS = self.sourceEOS = self.sourcePAD = self.sourceUNK = 0
        self.targetSOS = self.targetEOS = self.targetPAD = 0
        self.corruption_prob=p
        self.sentenceNumber = sentence
        self.times=rep
        self.bpe=bpe
        self.bpe_tokenizer=None

    def loadDictionaries(self, sentence, rep, p, timestamp):
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
        if self.bpe==0:
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

        else:
            index_to_word_dict = load_pickle(f'data/dictionaries/{sentence}_index_to_word_BPE.pkl',
                                             ["<sos>", "<eos>", "<pad>", "<unk>"],
                                             True)
            word_dict = load_pickle(f'data/dictionaries/{sentence}_word_to_index_BPE.pkl',
                                    ["<sos>", "<eos>", "<pad>", "<unk>"], False)
            target_word_dict = load_pickle(f'data/dictionaries/{sentence}_target_word_to_index_BPE.pkl',
                                           ["<sos>", "<eos>", "<pad>"], False)
            index_to_target_word_dict = load_pickle(f'data/dictionaries/{sentence}_index_to_target_word_BPE.pkl',
                                                    ["<sos>", "<eos>", "<pad>"], True)
            self.bpe_tokenizer=Tokenizer.from_file(f"data/dictionaries/bpe_tokenizer_{self.sentenceNumber}x{self.times}x{self.corruption_prob}.json")
            self.sourcePAD = self.bpe_tokenizer.token_to_id("<pad>")
            self.sourceUNK = self.bpe_tokenizer.token_to_id("<unk>")
            self.sourceSOS = self.bpe_tokenizer.token_to_id("<sos>")
            self.sourceEOS = self.bpe_tokenizer.token_to_id("<eos>")

            self.targetPAD = self.bpe_tokenizer.token_to_id("<pad>")
            self.targetUNK = self.bpe_tokenizer.token_to_id("<unk>")
            self.targetSOS = self.bpe_tokenizer.token_to_id("<sos>")
            self.targetEOS = self.bpe_tokenizer.token_to_id("<eos>")

        try:
            if self.bpe==0:
                with open(f'data/dictionaries/{sentence}x{rep}x{p}_SentenceMap.pkl', 'rb') as f:
                    sentenceMap = pickle.load(f)
                tqdm.write("Sentence map has been loaded.")
            else:
                with open(f'data/dictionaries/{sentence}x{rep}x{p}_SentenceMap_BPE.pkl', 'rb') as f:
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
                if random.random() < corruption_prob:
                    if (len(word)>1):
                        corruption_type = random.choice(['add', 'delete', 'change'])
                    else:
                        corruption_type = random.choice(['add', 'change'])
                    if char in string.ascii_letters:
                        if corruption_type == 'add':
                            corrupted_word.append(char)
                            corrupted_word.append(random.choice(keyboard_neighbors[char]))
                        elif corruption_type == 'delete':
                            continue
                        elif corruption_type == 'change':
                            corrupted_word.append(random.choice(keyboard_neighbors[char]))
                    # numeri
                    elif char.isdigit():
                        corrupted_word.append(char)
                            # punteggiatura
                    else:
                        if corruption_type == 'delete':
                            continue
                else:
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

        results = []
        for _ in range(times):
            if random.random() < 0.1:
                # ✅ mantieni la frase intatta
                results.append(' '.join(words))
            else:
                # 🔁 corrompi parola per parola
                corrupted = [self.corrupt_word_multiple(word, corruption_prob) for word in words]
                results.append(' '.join(corrupted))
        return results

    def buildDictionary(self, timestamp):
        """
        Reads a dataset, tokenizes sentences, and builds word dictionaries.
        """
        if self.bpe==0:
            if os.path.exists("data/tokenized_sentences"):
                tqdm.write("Loading pre-tokenized sentences...")
                with open("data/tokenized_sentences", "rb") as f:
                    sentences = pickle.load(f)
            else:
                tqdm.write("Reading file...")
                with open("data/WikiArticlesCorrect", "r", encoding="utf-16") as f:
                    article = f.read()
                tqdm.write("Tokenizing sentences...")
                sentences = re.split(r'[.!?]', article)

                with open("data/tokenized_sentences", "wb") as f:
                    pickle.dump(sentences, f)


            all_words = []
            all_target_words=[]
            all_sentences = {}

            for sentence in tqdm(sentences[:self.sentenceNumber], desc="Processing sentences"):
                #tqdm.write(f"Processing sentence {counter + 1}/{len(sentences)} ({(counter + 1) / len(sentences) * 100:.2f}%)")
                words = [word for word in word_tokenize(sentence) if word not in string.punctuation]
                finalSentence = ' '.join(words)
                all_words.extend(words, "'")
                all_target_words.extend(words, "'")


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
        else:
            tqdm.write("Preparing BPE tokenizer...")

            # 🔹 1. Splitting in sentences (. ! ?)
            with open("data/WikiArticlesCorrect", "r", encoding="utf-16") as f:
                article = f.read()
            sentences = re.split(r'[.!?]', article)

            all_texts = []
            for sentence in tqdm(sentences[:self.sentenceNumber], desc="Collecting texts for BPE"):
                words = [w for w in sentence.split() if w not in string.punctuation]
                if words:
                    all_texts.append(' '.join(words))

            # 🔹 2. Init BPE tokenizer
            tokenizer = Tokenizer(models.BPE())
            tokenizer.normalizer = normalizers.NFKC()
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

            # 🔹 3. Trainer (vocab size controlla granularità)
            trainer = trainers.BpeTrainer(vocab_size=30000, special_tokens=["<sos>", "<eos>", "<pad>", "<unk>"])

            # 🔹 4. Train
            tokenizer.train_from_iterator(all_texts, trainer)



            # 🔹 6. Ottieni vocabolario
            vocab = tokenizer.get_vocab()
            word_to_index = {word: idx for word, idx in vocab.items()}
            index_to_word = {idx: word for word, idx in vocab.items()}

            # per semplicità target = stesso vocabolario
            target_word_to_index = word_to_index
            index_to_target_word = index_to_word

            # 🔹 7. Genera il SentenceMap (corrupted → originale)
            all_sentences = {}
            for sentence in tqdm(all_texts, desc="Corrupting with BPE"):
                words = sentence.split()
                for corrupted_sentence in self.corrupt_sentence(words):
                    all_sentences[corrupted_sentence] = sentence

            # 🔹 8. Salvataggi
            with open(f'data/dictionaries/{self.sentenceNumber}_word_to_index_BPE.pkl', 'wb') as f:
                pickle.dump(word_to_index, f)

            with open(f'data/dictionaries/{self.sentenceNumber}_target_word_to_index_BPE.pkl', 'wb') as f:
                pickle.dump(target_word_to_index, f)

            with open(f'data/dictionaries/{self.sentenceNumber}_index_to_target_word_BPE.pkl', 'wb') as f:
                pickle.dump(index_to_target_word, f)

            with open(f'data/dictionaries/{self.sentenceNumber}_index_to_word_BPE.pkl', 'wb') as f:
                pickle.dump(index_to_word, f)

            with open(f'data/dictionaries/{self.sentenceNumber}x{self.times}x{self.corruption_prob}_SentenceMap_BPE.pkl',
                      'wb') as f:
                pickle.dump(all_sentences, f)

            # 🔹 9. Salva anche il tokenizer BPE per uso futuro
            tokenizer.save(f"data/dictionaries/bpe_tokenizer_{self.sentenceNumber}x{self.times}x{self.corruption_prob}.json")

            tqdm.write("BPE tokenizer and dictionaries saved successfully.")


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