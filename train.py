import gc
import multiprocessing
import os
import time
from datetime import datetime
import random

import torch
from sacremoses import MosesTokenizer
from torch.utils.data import DataLoader
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter

from DataLoader import TranslationDataset


def train(model, optimizer, scheduler, train_data, builder, word_dict, renormalizationLimit, maximumlearningRateLimit,
          target_word_dict, batch_size=64):
    print("Training started.")
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")  # Standard loss, no need to ignore padding
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    tokenizer = MosesTokenizer('en')
    global_step = 0

    # Create dataset
    dataset = TranslationDataset(train_data, word_dict,target_word_dict, builder, tokenizer)
    print("Dataset created")
    # Create batches of equal length sequences
    batch_sampler = create_memory_aware_batch_sampler(dataset, batch_size)
    # Check number of CPUs
    print("Batch sampler created")
    cpu_count = multiprocessing.cpu_count()

    # Check if CUDA is available
    is_cuda = torch.cuda.is_available()
    timestamp = str(int(time.time()))
    print("Num of CPU: ", cpu_count)
    print("GPU available: ", is_cuda)
    print("Timestamp: ", timestamp)

    # Convert it to a string

    # Decide number of workers
    # Decide number of workers
    if is_cuda:
        # On GPU: use more workers, but not more than available CPUs
        workers = min(8, cpu_count)
        log_dir = "/content/drive/MyDrive/runs/"+ timestamp
        writer = SummaryWriter(log_dir=log_dir)
    else:
        # On CPU: use fewer workers
        workers = min(4, cpu_count // 2)
        writer = SummaryWriter()

    # DataLoader
    train_loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_equal_length_fn,
        num_workers=workers,
        pin_memory=is_cuda  # Only pin memory if using GPU
    )
    counter=1
    while optimizer.param_groups[0]['lr'] > maximumlearningRateLimit:
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")
        epoch_loss = 0.0

        # Iterate through batches
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            gc.collect()
            torch.cuda.empty_cache()

            source = batch['source'].to(model.device)
            target = batch['target'].to(model.device)

            # Forward pass with batch
            predictions, logits = model(source, target)

            # Calculate loss
            # Reshape logits to match CrossEntropyLoss expectations
            # Assuming logits has shape [batch_size, seq_len, vocab_size]
            batch_size, seq_len, target_vocab_size = logits.size()
            logits_flat = logits.reshape(batch_size * seq_len, target_vocab_size)
            target_flat = target.reshape(batch_size * seq_len)

            loss = loss_fn(logits_flat, target_flat)
            writer.add_scalar('Loss/train', loss.item(), global_step=global_step)
            print(
                f"Batch {batch_idx}, Loss: {loss.item()}, Batch size: {source.size(0)}, Sequence length: {source.size(1)}")
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=renormalizationLimit)


            optimizer.step()
            epoch_loss += loss.item()
            global_step += 1
            del source, target, loss


        print(f"Epoch {counter} finished, average loss: {epoch_loss / len(train_loader)}")
        # Step the scheduler
        if counter>=5:
            scheduler.step()
        counter+=1


def tokenizeSentence(input_sentence):
    mt = MosesTokenizer('en')
    return mt.tokenize(input_sentence)


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


def create_memory_aware_batch_sampler(dataset, batch_size=64):
    """
    Creates a batch sampler with more aggressive memory constraints
    """
    # Make token limit very conservative
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        free_memory = total_memory - torch.cuda.memory_allocated(device)

        # Very conservative estimate (20% of free memory)
        token_size_bytes = 64  # Higher estimate considering optimizer states
        max_tokens = int((free_memory * 0.2) / token_size_bytes)

        # Hard cap to avoid large batches
        max_tokens = min(max_tokens, 2048)
    else:
        max_tokens = 1024  # Conservative default for CPU

    print(f"Using maximum of {max_tokens} tokens per batch based on available memory")

    # Group indices by source sequence length
    indices_by_length = {}
    for idx in range(len(dataset)):
        source_len = len(dataset.data[idx][0])
        if source_len not in indices_by_length:
            indices_by_length[source_len] = []
        indices_by_length[source_len].append(idx)

    # Create batches of equal length sequences
    batches = []
    for length, indices in sorted(indices_by_length.items()):
        # Calculate how many sequences we can fit in a batch
        tokens_per_seq = length * 4  # Account for source, target, and optimizer states
        max_seqs_per_batch = min(batch_size, max_tokens // tokens_per_seq)

        # Safety: ensure batch size is at least 1 but no more than 8
        max_seqs_per_batch = max(1, min(max_seqs_per_batch, batch_size))

        for i in range(0, len(indices), max_seqs_per_batch):
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