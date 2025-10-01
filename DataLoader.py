import random
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class TranslationDataset(Dataset):
    """
    A custom PyTorch Dataset for handling corrupted (source) and original (target) sentence pairs.

    This class takes a dictionary of sentence pairs, converts them into numerical indices based on
    a provided vocabulary, and stores them for efficient loading during model training. It supports
    both word-level and BPE (Byte-Pair Encoding) tokenization.
    """

    def __init__(self, data_dict, word_dict, target_word_dict, builder, tokenize_fn, fixedNumberofInputElements):
        """
            Initializes and preprocesses the dataset.

            Args:
                data_dict (dict): A dictionary mapping corrupted sentences (source) to original sentences (target).
                word_dict (dict): A vocabulary mapping source words/tokens to their integer indices.
                target_word_dict (dict): A vocabulary mapping target words/tokens to their integer indices.
                builder (BuildDictionary_Map): An instance of the data builder class, used to access special tokens and BPE tokenizer.
                tokenize_fn: A tokenizer function (though the logic now assumes pre-tokenized input).
                fixedNumberofInputElements (int): The maximum allowed sequence length for a pair to be included in the dataset.
        """

        self.data = []
        self.tokenize_fn = tokenize_fn
        self.word_dict = word_dict
        self.target_word_dict=target_word_dict
        self.builder = builder
        tqdm.write("Dictionary build up Started")
        # Iterate over each source (corrupted) and target (original) pair in the input dictionary.
        for idx, (source, target) in enumerate(tqdm(data_dict.items(), desc="Tokenizing data")):

            # --- Word-Level Tokenization Branch ---
            if builder.bpe==0:
                source_tokens = source
                target_tokens = target
                skip=False
                target_indices = [builder.targetSOS]

                # Convert target tokens to indices, starting with the Start-Of-Sentence token.
                for token in target_tokens:
                    idx = target_word_dict.get(token)
                    # If a token is not in the target vocabulary, flag this pair for skipping.
                    if idx is None:
                        tqdm.write(f"[DEBUG] Missing in target_word_dict: '{token}' | target: {target}")
                        skip = True
                        break
                    target_indices.append(idx)
                if skip:
                    continue

                # Convert source tokens to indices.
                source_indices = []
                for token in source_tokens:
                    idx = word_dict.get(token)
                    if idx is None:
                        tqdm.write(f"[DEBUG] Missing in word_dict: '{token}' | source: {source}")
                        source_indices.append(builder.sourceUNK)
                    else:
                        source_indices.append(idx)
                source_indices.append(builder.sourceEOS)

                # Filter the data: only include pairs where source and target have the same length and are within the maximum allowed length
                if (len(source_indices)==len(target_indices) and len(source_indices) <= fixedNumberofInputElements):
                    self.data.append((source_indices, target_indices))

            # --- BPE (Byte-Pair Encoding) Tokenization Branch ---
            else:
                print(source)
                print(target)
                source_text = " ".join(source)
                target_text = " ".join(target)

                # Use the trained BPE tokenizer to encode the strings into sequences of IDs.
                source_enc = builder.bpe_tokenizer.encode(source_text)
                target_enc = builder.bpe_tokenizer.encode(target_text)
                                
                # Get the lists of token IDs and add special tokens.
                source_indices = source_enc.ids + [builder.sourceEOS]
                target_indices = [builder.targetSOS] + target_enc.ids

                # Filter the data to include only pairs within the specified length limits.
                if (len(source_indices) < fixedNumberofInputElements-1 and len(target_indices) < fixedNumberofInputElements-1):
                    self.data.append((source_indices, target_indices))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_equal_length_fn(self, batch):
        """
        Custom collate function to process a batch of data points.

        For word-level tokenization, it assumes all sequences in the batch have the same length.
        For BPE, it pads sequences to the maximum length within the batch.

        Args:
            batch (list): A list of (source, target) tuples from the dataset.

        Returns:
            dict: A dictionary containing batched 'source' and 'target' tensors.
        """

        # Unzip the batch into separate lists for source and target sequences.
        source_seqs, target_seqs = zip(*batch)

        # Handle the word-level case (no padding needed due to batching strategy).
        if self.builder.bpe==0:
            # Convert to tensors
            source_tensor = torch.LongTensor(source_seqs)
            target_tensor = torch.LongTensor(target_seqs)

        # Handle the BPE case, which requires padding.
        else:
            # Get the padding token IDs from the builder.
            source_pad_id = self.builder.sourcePAD
            target_pad_id = self.builder.targetPAD

            # Find the maximum sequence length in the current batch for both source and target.
            max_src_len = max(len(seq) for seq in source_seqs)
            max_tgt_len = max(len(seq) for seq in target_seqs)

            # Pad each source sequence to the max source length.
            source_padded = [
                seq + [source_pad_id] * (max_src_len - len(seq))
                for seq in source_seqs
            ]
            # Pad each target sequence to the max target length.
            target_padded = [
                seq + [target_pad_id] * (max_tgt_len - len(seq))
                for seq in target_seqs
            ]

            # Convert the padded lists into LongTensors.
            source_tensor = torch.LongTensor(source_padded)
            target_tensor = torch.LongTensor(target_padded)

        return {
            'source': source_tensor,
            'target': target_tensor
        }

def create_equal_length_batches(dataset, fixedNumberOfInputElements, batch_size=64):
    """
    Groups data indices by source sequence length and creates batches.

    This strategy creates batches where every source sequence has the exact same length,
    eliminating the need for padding and potentially speeding up training for certain models.

    Args:
        dataset (TranslationDataset): The dataset to create batches from.
        fixedNumberOfInputElements (int): The maximum sequence length to consider.
        batch_size (int): The desired number of sequences per batch.

    Returns:
        list: A list of batches, where each batch is a list of indices.
    """
    # Group indices by source sequence length
    indices_by_length = {}

    # First pass: group indices of all data points by their source length.
    for idx in tqdm(range(len(dataset)), desc="Creating length-based groups"):
        # Get the length of the source sequence for the current data point.
        source_len = len(dataset.data[idx][0])

        # Filter out sequences that are too long or too short.
        if source_len > fixedNumberOfInputElements:
            continue
        if source_len < 4:
            continue

        # If this length hasn't been seen before, initialize a new list and append the list.
        if source_len not in indices_by_length:
            indices_by_length[source_len] = []
        indices_by_length[source_len].append(idx)



    # Create batches of equal length sequences
    batches = []
    total_groups = len(indices_by_length)
    current_group = 0

    # Second pass: create mini-batches from the length-grouped indices.
    for length, indices in tqdm(indices_by_length.items(), desc="Creating batches"):
        current_group += 1
        # Iterate through the list of indices for the current length in chunks of `batch_size`.
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            if batch_indices:  # Ensure batch is not empty
                batches.append(batch_indices)

    # Shuffle the batches
    random.shuffle(batches)
    return batches


