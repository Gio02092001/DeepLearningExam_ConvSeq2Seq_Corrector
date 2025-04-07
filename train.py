import multiprocessing
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
    writer = SummaryWriter()
    tokenizer = MosesTokenizer('en')
    global_step = 0

    # Create dataset
    dataset = TranslationDataset(train_data, word_dict,target_word_dict, builder, tokenizer)

    # Create batches of equal length sequences
    batch_sampler = create_equal_length_batches(dataset, batch_size)
    # Check number of CPUs
    cpu_count = multiprocessing.cpu_count()

    # Check if CUDA is available
    is_cuda = torch.cuda.is_available()

    # Decide number of workers
    if is_cuda:
        # On GPU: use more workers, but not more than available CPUs
        workers = min(8, cpu_count)
    else:
        # On CPU: use fewer workers
        workers = min(4, cpu_count // 2)

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