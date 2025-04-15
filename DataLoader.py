import random

import torch
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    """Custom dataset for translation pairs"""

    def __init__(self, data_dict, word_dict, target_word_dict, builder, tokenize_fn):
        self.data = []
        self.tokenize_fn = tokenize_fn
        self.word_dict = word_dict
        self.target_word_dict=target_word_dict
        self.builder = builder

        # Preprocess all data
        for source, target in data_dict.items():
            # Tokenize
            source_tokens = self.tokenize_fn.tokenize(source)
            target_tokens = self.tokenize_fn.tokenize(target)

            # Convert to indices
            source_indices = [word_dict.get(token) for token in source_tokens] + [builder.sourceEOS]
            target_indices = [builder.targetSOS] + [target_word_dict.get(token) for token in target_tokens]
            self.data.append((source_indices, target_indices))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]




def create_memory_aware_batch_sampler(dataset, batch_size=64):
    """
    Creates a batch sampler with more aggressive memory constraints
    """
    # Group indices by source sequence length
    indices_by_length = {}
    for idx in range(len(dataset)):
        source_len = len(dataset.data[idx][0])
        if source_len not in indices_by_length:
            indices_by_length[source_len] = []
        indices_by_length[source_len].append(idx)
    # Make token limit very conservative
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        free_memory = total_memory - torch.cuda.memory_allocated(device)

        # Very conservative estimate (20% of free memory)
        token_size_bytes = 16  # Higher estimate considering optimizer states
        max_tokens = int((free_memory * 0.9) / token_size_bytes)
        print(f"Using maximum of {max_tokens} tokens per batch based on available memory on CUDA")


    # Create batches of equal length sequences
    batches = []
    for length, indices in sorted(indices_by_length.items()):
        # Calculate how many sequences we can fit in a batch
        if torch.cuda.is_available():
            tokens_per_seq = length * 4  # Account for source, target, and optimizer states
            max_seqs_per_batch = min(batch_size, max_tokens // tokens_per_seq)

            # Safety: ensure batch size is at least 1 but no more than 8
            max_seqs_per_batch = max(1, min(max_seqs_per_batch, batch_size))
        else:
            max_seqs_per_batch=batch_size

        for i in range(0, len(indices)):
            batch_indices = indices[i:i + max_seqs_per_batch]
            if batch_indices:  # Ensure batch is not empty
                batches.append(batch_indices)

    # Shuffle the batches
    random.shuffle(batches)
    return batches

def collate_equal_length_fn(batch):
    """
    Custom collate function for batches of equal length sequences.
    No padding required since all sequences in a batch already have the same length.
    """
    # Separate source and target sequences
    source_seqs, target_seqs = zip(*batch)

    # Convert to tensors
    source_tensor = torch.LongTensor(source_seqs)
    target_tensor = torch.LongTensor(target_seqs)

    return {
        'source': source_tensor,
        'target': target_tensor
    }