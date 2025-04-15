import gc
import multiprocessing
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from DataLoader import TranslationDataset, create_memory_aware_batch_sampler, collate_equal_length_fn


def validation(validation_data, model, tokenizer, word_dict, target_word_dict, builder, batch_size=64):
    print("Validation started.")
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()  # Standard loss, no need to ignore padding
    global_step = 0

    # Create dataset
    dataset = TranslationDataset(validation_data, word_dict, target_word_dict, builder, tokenizer)
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
        log_dir = "/content/drive/MyDrive/runs/" + timestamp
        writer = SummaryWriter(log_dir=log_dir)
    else:
        # On CPU: use fewer workers
        workers = min(4, cpu_count // 2)
        writer = SummaryWriter()

    # DataLoader
    validationLoader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_equal_length_fn,
        num_workers=workers,
        pin_memory=is_cuda  # Only pin memory if using GPU
    )
    counter = 1

    epoch_loss = 0.0
    epoch_perplexity = 0.0
    total_tokens = 0
    correct_tokens = 0
    total_edit_distance = 0
    num_sequences = 0

    # Iterate through batches
    for batch_idx, batch in enumerate(validationLoader):
        gc.collect()
        torch.cuda.empty_cache()

        source = batch['source'].to(model.device)
        target = batch['target'].to(model.device)  # If you want to compare later
        sourceBatchSize = source.shape[0]

        # Start with only SOS tokens
        targetInput = torch.full((sourceBatchSize, 1), builder.targetSOS, dtype=torch.long, device=model.device)

        # Placeholder to collect predictions
        predictedSequence = []

        for _ in range(source.shape[1]):  # You should define this limit!
            predictions, logits = model(source, targetInput)

            # Get the last token logits
            next_tokens = torch.argmax(logits[:, -1, :], dim=1, keepdim=True)  # Shape: (batch_size, 1)

            # Append predicted token to targetInput
            targetInput = torch.cat([targetInput, next_tokens], dim=1)

            predictedSequence.append(next_tokens)

        # After the loop, stack the sequence
        predictedSequence = torch.cat(predictedSequence, dim=1)  # Shape: (batch_size, max_output_length)
        print(predictedSequence)

        """POI QUANDO QUESTO FUNZIONA CAMBIA ARGMAX PER BEAM SEARCH"""
        # Calculate loss
        # Reshape logits to match CrossEntropyLoss expectations
        # Assuming logits has shape [batch_size, seq_len, vocab_size]
        batch_size, seq_len, target_vocab_size = logits.size()
        logits_flat = logits.reshape(batch_size * seq_len, target_vocab_size)
        target_flat = target.reshape(batch_size * seq_len)

        loss = loss_fn(logits_flat, target_flat)
        perplexity = torch.exp(loss)
        writer.add_scalar('Loss/validation', loss.item(), global_step=global_step)
        writer.add_scalar('Perplexity/validation', perplexity, global_step=global_step)
        print(
            f"VALIDATION Batch {batch_idx}, Loss: {loss.item()}, Perplexity: {perplexity}, Batch size: {source.size(0)}, Sequence length: {source.size(1)}")
        # Backward pass

        epoch_loss += loss.item()
        epoch_perplexity +=perplexity
        global_step += 1
        del source, target, logits, predictions, loss, logits_flat, target_flat
        torch.cuda.empty_cache()
        gc.collect()

    epoch_perplexity=epoch_perplexity/ len(validationLoader)

    print(f"Epoch {counter} finished, average loss: {epoch_loss / len(validationLoader)}")
    print(f"Epoch perplexity: {epoch_perplexity}")
    return epoch_perplexity

