import os
import pickle
import string
import random
import re
from nltk.tokenize import TweetTokenizer
import pandas as pd
from tqdm import tqdm
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors, normalizers

class BuildDictionary_Map:
    """
        Manages the creation, corruption, and loading of text datasets for a sequence-to-sequence model.

        This class handles the entire preprocessing pipeline, including:
        1. Reading a source text corpus.
        2. Tokenizing sentences.
        3. Artificially "corrupting" sentences to simulate typographical errors.
        4. Building source (corrupted) and target (original) vocabulary dictionaries.
        5. Creating a map from corrupted sentences to their original versions.
        6. Supporting both word-level and Byte-Pair Encoding (BPE) tokenization.
        7. Saving and loading these generated assets for later use in model training.
    """
    # Default parameters for data generation.
    tokenizer = TweetTokenizer(preserve_case=True)
    corruption_prob = 0.1
    times = 5
    sentenceNumber = 100000

    def __init__(self, sentence, rep, p, bpe):
        """
               Initializes the BuildDictionary_Map instance with specific configuration.

               Args:
                   sentence (int): The number of sentences to process from the source corpus.
                   rep (int): The number of times each sentence should be corrupted.
                   p (float): The probability of corruption for each character.
                   bpe (int): A flag to indicate whether to use Byte-Pair Encoding (1 for BPE, 0 for word-level).
                   timestamp (any): A timestamp or identifier for the run (not used in this class but kept for API consistency).
        """
        self.sourceSOS = self.sourceEOS = self.sourcePAD = self.sourceUNK = 0
        self.targetSOS = self.targetEOS = self.targetPAD = 0
        self.corruption_prob=p
        self.sentenceNumber = sentence
        self.times=rep
        self.bpe=bpe
        self.bpe_tokenizer=None

    def loadDictionaries(self, sentence, rep, p, timestamp):
        """
        Loads pre-computed dictionaries and the sentence map from disk.

        This function checks for existing .pkl files based on the dataset parameters.
        It handles loading for both BPE and word-level tokenized data.

        Args:
            sentence (int): The number of sentences used to generate the dictionaries.
            rep (int): The repetition factor used.
            p (float): The corruption probability used.
            timestamp (any): An identifier for the run (not used here).

        Returns:
            tuple: A tuple containing:
                - word_dict (dict): Mapping from source words/tokens to indices.
                - target_word_dict (dict): Mapping from target words/tokens to indices.
                - sentenceMap (dict): Mapping from corrupted sentences to original sentences.
                - index_to_target_word_dict (dict): Mapping from target indices to words/tokens.
                - index_to_word_dict (dict): Mapping from source indices to words/tokens.
        """

        def load_pickle(filename, special_tokens,index_to_Target):
            """Helper function to load a pickle file and add special tokens."""
            try:
                with open(filename, 'rb') as f:
                    dictionary = pickle.load(f)

                    # Add special tokens to the loaded dictionary.
                    for token in special_tokens:
                        if index_to_Target:
                            dictionary[len(dictionary) + 1]=token
                        else:
                            dictionary[token] = len(dictionary) + 1
                    return dictionary
            except FileNotFoundError:
                tqdm.write(f"The file '{filename}' was not found.")
                return {}

        # Check if BPE is disabled (using word-level tokenization).
        if self.bpe==0:

            # Load the four dictionaries: index-to-word, word-to-index for both source and target.
            index_to_word_dict = load_pickle(f'data/dictionaries/{sentence}_index_to_word.pkl', ["<sos>", "<eos>", "<pad>", "<unk>"],
                                    True)
            word_dict = load_pickle(f'data/dictionaries/{sentence}_word_to_index.pkl', ["<sos>", "<eos>", "<pad>", "<unk>"], False)
            target_word_dict = load_pickle(f'data/dictionaries/{sentence}_target_word_to_index.pkl',
                                           ["<sos>", "<eos>", "<pad>"], False)
            index_to_target_word_dict=load_pickle(f'data/dictionaries/{sentence}_index_to_target_word.pkl',
                                                  ["<sos>", "<eos>", "<pad>"],True)

            # Assign the indices of special tokens from the loaded dictionary.
            self.sourceSOS = word_dict["<sos>"]
            self.sourceEOS = word_dict["<eos>"]
            self.sourcePAD = word_dict["<pad>"]
            self.sourceUNK = word_dict["<unk>"]
            self.targetSOS = target_word_dict["<sos>"]
            self.targetEOS = target_word_dict["<eos>"]
            self.targetPAD = target_word_dict["<pad>"]

        # If BPE is enabled.
        else:
            # Load the dictionaries and Tokenizer
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

            # Assign special toknes
            self.sourcePAD = self.bpe_tokenizer.token_to_id("<pad>")
            self.sourceUNK = self.bpe_tokenizer.token_to_id("<unk>")
            self.sourceSOS = self.bpe_tokenizer.token_to_id("<sos>")
            self.sourceEOS = self.bpe_tokenizer.token_to_id("<eos>")

            self.targetPAD = self.bpe_tokenizer.token_to_id("<pad>")
            self.targetUNK = self.bpe_tokenizer.token_to_id("<unk>")
            self.targetSOS = self.bpe_tokenizer.token_to_id("<sos>")
            self.targetEOS = self.bpe_tokenizer.token_to_id("<eos>")

        # Attempt to load the sentence map (corrupted -> original).
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
        Randomly corrupts a single word by adding, deleting, or changing characters.

        This function simulates common typing errors by modifying characters based on
        their neighbors on a standard QWERTY keyboard layout.

        Args:
            word (str): The word to corrupt.
            corruption_prob (float, optional): The probability of corrupting a character.
                                               Defaults to the instance's `self.corruption_prob`.

        Returns:
            str: The corrupted word. Can be an empty string if all characters are deleted.
        """

        # Define set of special tokens that should never be corrupted.
        SPECIAL_TOKENS = {"<UNK>", "<unk>", "<eos>", "<sos>", "<pad>"}
        if word in SPECIAL_TOKENS:
            return word
        else:

            # A dictionary mapping each character to its neighbors on a QWERTY keyboard.
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

            # Iterate over each character in the input word.
            for char in word:
                # Decide whether to corrupt the character based on the probability
                if random.random() < corruption_prob:

                    # For words with more than one character, allow deletion.
                    if (len(word)>1):
                        corruption_type = random.choice(['add', 'delete', 'change'])
                    else:
                        corruption_type = random.choice(['add', 'change'])

                    # Check if the character is an alphabet letter.
                    if char in string.ascii_letters:
                        if corruption_type == 'add':
                            corrupted_word.append(char)
                            corrupted_word.append(random.choice(keyboard_neighbors[char]))
                        elif corruption_type == 'delete':
                            continue
                        elif corruption_type == 'change':
                            corrupted_word.append(random.choice(keyboard_neighbors[char]))

                    # Handle numbers (currently only keeps the number).
                    elif char.isdigit():
                        corrupted_word.append(char)

                    # Handle punctuation and other symbols.
                    else:
                        if corruption_type == 'delete':
                            continue
                else:
                    corrupted_word.append(char)

            # Apply a final corruption check to potentially add a random character at the end.
            if random.random() < corruption_prob:
                corrupted_word.append(random.choice(string.ascii_letters))

            # Join the list of characters back into a string.
            if len(''.join(corrupted_word))<1:
                return random.choice(string.ascii_letters)
            else:
                return ''.join(corrupted_word)

    def corrupt_sentence(self, words, corruption_prob=None, times=None):
        """
        Generates multiple corrupted versions of a sentence.

        Args:
            words (list[str]): A list of words representing the original sentence.
            corruption_prob (float, optional): The corruption probability. Defaults to instance default.
            times (int, optional): The number of corrupted versions to generate. Defaults to instance default.

        Returns:
            list[list[str]]: A list containing multiple corrupted versions of the sentence.
        """
        corruption_prob = corruption_prob or self.corruption_prob
        times = times or self.times

        results = []
        # Loop to create the specified number of corrupted versions.
        for _ in range(times):
            # With a probability of `1 - corruption_prob`, keep the sentence unchanged.
            if random.random() > corruption_prob:
                # ✅ mantieni la frase intatta
                results.append(words[:])
            else:
                # 🔁 corrompi parola per parola
                corrupted = [self.corrupt_word_multiple(word, corruption_prob) for word in words]
                results.append(corrupted)
        return results

    def buildDictionary(self):
        """
        Constructs the dictionaries and sentence map from a source corpus.

        This is the main data generation method. It reads a large text file,
        processes a specified number of sentences, corrupts them, and builds the
        necessary vocabulary and mapping files for the model. It supports both
        word-level and BPE tokenization.

        """
        # --- Word-Level Tokenization Branch ---
        if self.bpe==0:
            # Check if tokenized sentences have been pre-computed and saved.
            if os.path.exists("data/tokenized_sentences"):
                tqdm.write("Loading pre-tokenized sentences...")
                with open("data/tokenized_sentences_full", "rb") as f:
                    sentences = pickle.load(f)
            else:
                tqdm.write("Reading file...")
                with open("data/WikiArticlesCorrect", "r", encoding="utf-16") as f:
                    article = f.read()
                tqdm.write("Tokenizing sentences...")
                sentences = re.split(r'[.!?]', article)
                tqdm.write("Split Done")

                # Tokenize each sentence into words using the NLTK TweetTokenizer.
                sentences = self.tokenizer.tokenize_sents(sentences)
                with open("data/tokenized_sentences_full", "wb") as f:
                    pickle.dump(sentences, f)

            all_words = []
            all_target_words=[]
            all_sentences = {}

            # Process the specified number of sentences.
            for sentence in tqdm(sentences[:self.sentenceNumber], desc="Processing sentences"):
                # Add original words to both source and target vocabularies.
                all_words.extend(sentence)
                all_target_words.extend(sentence)

                # Generate corrupted versions of the current sentence.
                for corrupted_sentence in self.corrupt_sentence(sentence):
                    all_words.extend(corrupted_sentence)
                    all_sentences[tuple(corrupted_sentence)] = sentence

            # Create dictionaries by finding unique words and assigning indices.
            index_to_word = {idx: word for idx, word in enumerate(pd.Series(all_words).drop_duplicates())}
            word_to_index = {word: idx for idx, word in enumerate(pd.Series(all_words).drop_duplicates())}
            target_word_to_index = {word: idx for idx, word in enumerate(pd.Series(all_target_words).drop_duplicates())}
            index_to_target_word = {idx: word for idx, word in enumerate(pd.Series(all_target_words).drop_duplicates())}

            # Save all generated dictionaries and the sentence map to pickle files.
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

        # --- BPE Tokenization Branch ---
        else:
            tqdm.write("Preparing BPE tokenizer...")

            # Step 1: Read the corpus and split it into sentences.
            with open("data/WikiArticlesCorrect", "r", encoding="utf-16") as f:
                article = f.read()
            sentences = re.split(r'[.!?]', article)

            # Collect sentences as strings for the BPE trainer.
            all_texts = []
            for sentence in tqdm(sentences[:self.sentenceNumber], desc="Collecting texts for BPE"):
                words = [w for w in sentence.split()]
                if words:
                    all_texts.append(' '.join(words))

            # Step 2: Initialize a BPE tokenizer model.
            tokenizer = Tokenizer(models.BPE())
            tokenizer.normalizer = normalizers.NFKC()
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

            # Step 3: Configure the trainer.
            trainer = trainers.BpeTrainer(vocab_size=30000, special_tokens=["<sos>", "<eos>", "<pad>", "<unk>"])

            # Step 4: Train the tokenizer on the collected text data.
            tokenizer.train_from_iterator(all_texts, trainer)

            # Step 5: Extract the vocabulary from the trained tokenizer.
            vocab = tokenizer.get_vocab()
            word_to_index = {word: idx for word, idx in vocab.items()}
            index_to_word = {idx: word for word, idx in vocab.items()}
            print(index_to_word[1000], "+", index_to_word[1100])
            target_word_to_index = word_to_index
            index_to_target_word = index_to_word
            print(index_to_target_word[1000], "+", index_to_target_word[1100])

            # Step 6: Generate the sentence map (corrupted -> original).
            all_sentences = {}
            for sentence in tqdm(all_texts, desc="Corrupting with BPE"):
                words = sentence.split()
                for corrupted_sentence in self.corrupt_sentence(words):
                    all_sentences[tuple(corrupted_sentence)] = sentence

            # Step 7: Save the BPE dictionaries and sentence map.
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

            # Step 8: Save the configured and trained tokenizer object itself for later use.
            tokenizer.save(f"data/dictionaries/bpe_tokenizer_{self.sentenceNumber}x{self.times}x{self.corruption_prob}.json")
            tqdm.write("BPE tokenizer and dictionaries saved successfully.")


    def splitSet(self, sentenceMap, validationSet):
        """
        Splits the complete sentence map into training and validation sets.

        Args:
            sentenceMap (dict): The dictionary mapping corrupted to original sentences.
            validationSet (float): The proportion of the data to be used for validation (e.g., 0.1 for 10%).

        Returns:
            tuple: A tuple containing:
                - train_data (dict): The training set portion of the sentence map.
                - validation_data (dict): The validation set portion of the sentence map.
        """

        # Set a fixed seed for random sampling to ensure reproducibility.
        random.seed(42)

        # Calculate the number of samples for the validation set.
        val_size = int(validationSet * len(sentenceMap))

        # Randomly sample keys from the sentence map to be included in the validation set.
        val_keys = random.sample(list(sentenceMap.keys()), val_size)

        # Initialize dictionaries for the new splits.
        train_data = {}
        validation_data = {}

        # Iterate through the original sentence map to distribute items into the splits.
        for k, v in tqdm(sentenceMap.items(), desc="Splitting train/validation"):
            if k in val_keys:
                validation_data[k] = v
            else:
                train_data[k] = v

        tqdm.write("Completed splitting data!")
        tqdm.write(f"Train size: {len(train_data)}, Validation size: {len(validation_data)}")
        return train_data, validation_data