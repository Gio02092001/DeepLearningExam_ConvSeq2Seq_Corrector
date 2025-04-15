
def create_equal_length_batches(dataset, batch_size=64):
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

    for idx in range(len(dataset)):
        source_len = len(dataset.data[idx][0])
        if source_len not in indices_by_length:
            indices_by_length[source_len] = []
        indices_by_length[source_len].append(idx)

    # Create batches of equal length sequences
    batches = []
    for length, indices in indices_by_length.items():
        # Split into batches of batch_size
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            if batch_indices:  # Ensure batch is not empty
                batches.append(batch_indices)

    # Shuffle the batches
    random.shuffle(batches)
    return batches

def create_equal_length_batches(dataset, batch_size=64, max_tokens_per_batch=4096):
    """
    Create batches where all sequences within a batch have exactly the same length.
    Splits batches that would exceed memory limits.

    Args:
        dataset: The TranslationDataset
        batch_size: Maximum number of sequences per batch
        max_tokens_per_batch: Maximum number of tokens allowed in a single batch

    Returns:
        List of batch indices
    """
    # Group indices by source sequence length
    indices_by_length = {}
    for idx in range(len(dataset)):
        source_len = len(dataset.data[idx][0])
        if source_len not in indices_by_length:
            indices_by_length[source_len] = []
        indices_by_length[source_len].append(idx)

    # Create batches of equal length sequences
    batches = []
    for length, indices in indices_by_length.items():
        # Calculate how many sequences we can fit in a batch based on token limit
        # Each sequence has 'length' tokens
        max_seqs_per_batch = min(batch_size, max_tokens_per_batch // length)

        # If even a single sequence exceeds token limit, allow at least one sequence
        if max_seqs_per_batch == 0:
            max_seqs_per_batch = 1
            print(f"Warning: Sequence of length {length} exceeds max_tokens_per_batch. Processing one at a time.")

        # Split into batches with memory constraint
        for i in range(0, len(indices), max_seqs_per_batch):
            batch_indices = indices[i:i + max_seqs_per_batch]
            if batch_indices:  # Ensure batch is not empty
                batches.append(batch_indices)

    # Shuffle the batches
    random.shuffle(batches)
    return batches


def get_max_tokens_per_batch():
    """
    Determine the maximum number of tokens per batch based on available memory.
    Returns a conservative estimate that should work for most models.
    """
    if torch.cuda.is_available():
        # Get available GPU memory in bytes
        device = torch.cuda.current_device()
        available_memory = torch.cuda.get_device_properties(device).total_memory
        free_memory = available_memory - torch.cuda.memory_allocated(device)

        # Conservative estimate: use at most 80% of free memory
        # Assuming each token requires around 8-16 bytes in the worst case (depends on model)
        token_size_bytes = 16  # Conservative estimate for embeddings, gradients, etc.
        max_tokens = int((free_memory * 0.8) / token_size_bytes)

        # Cap at a reasonable maximum (adjust based on your model size)
        return min(max_tokens, 8192)
    else:
        # For CPU, use a more conservative default
        return 4096

