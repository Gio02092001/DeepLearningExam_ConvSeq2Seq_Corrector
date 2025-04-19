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
    startFineTuning = False
    while optimizer.param_groups[0]['lr'] > maximumlearningRateLimit:
        #validation_output= validation(validation_data, model, tokenizer, word_dict, target_word_dict, builder, fixedNumberOfInputElements, epochNumber, writer, batch_size)
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")
        epoch_loss = 0.0
        correct_tokens = 0  # Inizializza il contatore dei token corretti
        total_tokens = 0

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

            # Mask for excluding SOS and EOS tokens
            SOS_ID = builder.targetSOS
            EOS_ID = builder.targetEOS
            mask = (target_flat != SOS_ID) & (target_flat != EOS_ID)  # Exclude SOS and EOS tokens

            # Apply mask to logits and target
            logits_flat = logits_flat[mask]
            target_flat = target_flat[mask]

            loss = loss_fn(logits_flat, target_flat)
            writer.add_scalar('Loss/train', loss.item(), global_step=global_step)
            writer.flush()
            # Token-Level Accuracy
            predicted_tokens = torch.argmax(logits, dim=-1)

            correct_tokens_batch= (predicted_tokens == target).sum().item()# Predetti token per ogni sequenza
            correct_tokens += correct_tokens_batch  # Confronto token per token

            total_tokens += mask.sum().item() # Conta il numero totale di token nel batch
            accuracy_batch = correct_tokens_batch / mask.sum().item() if mask.sum().item() > 0 else 0.0
            writer.add_scalar('Accuracy/train', accuracy_batch, global_step=global_step)
            writer.flush()
            print(
                f"Batch {batch_idx}, Loss: {loss.item()}, Batch size: {source.size(0)}, Accuracy: {accuracy_batch}, Sequence length: {source.size(1)}")
            # Backward pass
            loss.backward()
            grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=renormalizationLimit)
            writer.add_scalar('GradNorm/train', grad_norm, global_step=global_step)
            writer.flush()

            optimizer.step()
            epoch_loss += loss.item()
            global_step += 1
            del source, target, logits, predictions, loss, logits_flat, target_flat
            torch.cuda.empty_cache()
            gc.collect()



        accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
        print(
            f"Epoch {epochNumber} finished, average loss: {epoch_loss / len(train_loader)}, Accuracy: {accuracy * 100:.2f}%")

        validation_output= validation(validation_data, model, tokenizer, word_dict, target_word_dict, builder, fixedNumberOfInputElements, epochNumber, writer, batch_size)

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
            best_validationOutput = validation_output
            print("first epoch completed")

        print("FineTuning started: ", startFineTuning)


        epochNumber+=1


def tokenizeSentence(input_sentence):
    mt = MosesTokenizer('en')
    return mt.tokenize(input_sentence)

