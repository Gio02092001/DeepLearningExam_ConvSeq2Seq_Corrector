import random

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class TranslationDataset(Dataset):
    """Custom dataset for translation pairs"""

    def __init__(self, data_dict, word_dict, target_word_dict, builder, tokenize_fn):
        self.data = []
        self.tokenize_fn = tokenize_fn
        self.word_dict = word_dict
        self.target_word_dict=target_word_dict
        self.builder = builder
        total = len(data_dict)
        tqdm.write("Dictionary build up Started")
        # Preprocess all data
        for idx, (source, target) in enumerate(tqdm(data_dict.items(), desc="Tokenizing data")):
            ''' quello che ho fatto nel train dovrei farlo qui perché devo farlo nel dataset a quel punto però posso togliere il 5. step perché tanto ce lo metto io qui già tutto.'''
            if builder.bpe==0:
                # Tokenize
                source_tokens = self.tokenize_fn.tokenize(source)
                target_tokens = self.tokenize_fn.tokenize(target)

                # Convert to indices
                source_indices = [word_dict.get(token) for token in source_tokens] + [builder.sourceEOS]
                target_indices = [builder.targetSOS] + [target_word_dict.get(token) for token in target_tokens]
                self.data.append((source_indices, target_indices))
            else:
                source_enc = builder.bpe_tokenizer.encode(source)
                target_enc = builder.bpe_tokenizer.encode(target)

                # Ottieni gli ID
                source_indices = source_enc.ids + [builder.sourceEOS]
                target_indices = [builder.targetSOS] + target_enc.ids

                # Salva la coppia (source, target)
                self.data.append((source_indices, target_indices))

            # tqdm.write progress every 0.1%

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_equal_length_fn(self, batch):
        """
        Custom collate function for batches of equal length sequences.
        No padding required since all sequences in a batch already have the same length.
        """
        # Separate source and target sequences
        source_seqs, target_seqs = zip(*batch)
        if self.builder.bpe==0:
            # Convert to tensors
            source_tensor = torch.LongTensor(source_seqs)
            target_tensor = torch.LongTensor(target_seqs)

        else:
             # Caso BPE → serve padding
            source_pad_id = self.builder.sourcePAD
            target_pad_id = self.builder.targetPAD

            # Trova la max lunghezza in batch
            max_src_len = max(len(seq) for seq in source_seqs)
            max_tgt_len = max(len(seq) for seq in target_seqs)

            # Pad source
            source_padded = [
                seq + [source_pad_id] * (max_src_len - len(seq))
                for seq in source_seqs
            ]
            # Pad target
            target_padded = [
                seq + [target_pad_id] * (max_tgt_len - len(seq))
                for seq in target_seqs
            ]

            source_tensor = torch.LongTensor(source_padded)
            target_tensor = torch.LongTensor(target_padded)

        return {
            'source': source_tensor,
            'target': target_tensor
        }



def create_equal_length_batches(dataset, fixedNumberOfInputElements, batch_size=64):
    """
    Create batches where all sequences within a batch have exactly the same length.
    No padding needed.

    Args:
        dataset: The TranslationDataset
        batch_size: Maximum number of sequences per batch

    Returns:
        List of batch indices
    """
    # Group indices by source sequence length
    indices_by_length = {}
    total = len(dataset)
    for idx in tqdm(range(len(dataset)), desc="Creating length-based groups"):

        source_len = len(dataset.data[idx][0])
        if source_len > fixedNumberOfInputElements:
            continue
        if source_len < 2:
            continue
        if source_len not in indices_by_length:
            indices_by_length[source_len] = []
        indices_by_length[source_len].append(idx)
        # tqdm.write progress every 0.1%


    # Create batches of equal length sequences
    batches = []
    total_groups = len(indices_by_length)
    current_group = 0

    for length, indices in tqdm(indices_by_length.items(), desc="Creating batches"):
        current_group += 1
        # Split into batches of batch_size
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            if batch_indices:  # Ensure batch is not empty
                batches.append(batch_indices)
        percent_group = (current_group / total_groups) * 100
        #tqdm.write(f"Completed {percent_group:.2f}% of sequence length groups.")

    # Shuffle the batches
    random.shuffle(batches)
    return batches


