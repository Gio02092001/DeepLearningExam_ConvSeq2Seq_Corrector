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

from DataLoader import TranslationDataset, collate_equal_length_fn, create_equal_length_batches
from validation import validation


def train(model, optimizer, scheduler, train_data, builder, word_dict, renormalizationLimit, maximumlearningRateLimit,
          target_word_dict,validation_data,fixedNumberOfInputElements, batch_size=64):
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
    batch_sampler = create_equal_length_batches(dataset,fixedNumberOfInputElements, batch_size)
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
    epochNumber=1
    best_validationOutput=0
    while optimizer.param_groups[0]['lr'] > maximumlearningRateLimit:
        #validation_output = validation(validation_data, model, tokenizer, word_dict, target_word_dict, builder)
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
            del source, target, logits, predictions, loss, logits_flat, target_flat
            torch.cuda.empty_cache()
            gc.collect()


        print(f"Epoch {epochNumber} finished, average loss: {epoch_loss / len(train_loader)}")
        validation_output= validation(validation_data, model, tokenizer, word_dict, target_word_dict, builder, fixedNumberOfInputElements)
        startFineTuning=False
        if epochNumber>1:
            if startFineTuning==False:
                if validation_output<best_validationOutput:
                    best_validationOutput=validation_output
                else:
                    scheduler.step()
                    startFineTuning=True
            else:
                scheduler.step()
        else:
            best_validationOutput=validation_output


        epochNumber+=1


def tokenizeSentence(input_sentence):
    mt = MosesTokenizer('en')
    return mt.tokenize(input_sentence)

