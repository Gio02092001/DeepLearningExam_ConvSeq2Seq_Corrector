import gc
import multiprocessing
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from DataLoader import TranslationDataset, create_equal_length_batches
from validation import validation

def train(model, optimizer, scheduler, train_data, builder, word_dict, renormalizationLimit, maximumlearningRateLimit,
          target_word_dict,validation_data,fixedNumberOfInputElements, batch_size, index_to_target_word_dict, patience, index_to_word_dict,timestamp, ckpt=None, pretrained=None):
    """
        The main function to handle the model training and validation loop.

        This function orchestrates the entire training process, including:
        - Setting up DataLoaders for training and validation sets.
        - Initializing TensorBoard for logging.
        - Handling resumption from checkpoints.
        - Iterating through epochs and batches.
        - Performing forward/backward passes and optimizer steps.
        - Applying on-the-fly data corruption during training.
        - Calling the validation function after each epoch.
        - Implementing an early stopping mechanism with patience.
        - Saving model checkpoints ('best_model.pt' and 'last_model.pt').
        - Adjusting the learning rate based on validation performance.

        Args:
            model (torch.nn.Module): The PyTorch model to be trained.
            optimizer (torch.optim.Optimizer): The optimizer for updating model weights.
            scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
            train_data (dict): The training dataset split.
            builder (BuildDictionary_Map): The data preparation and configuration object.
            word_dict (dict): The source vocabulary mapping words to indices.
            renormalizationLimit (float): The maximum norm for gradient clipping.
            maximumlearningRateLimit (float): The minimum learning rate threshold to stop training.
            target_word_dict (dict): The target vocabulary mapping words to indices.
            validation_data (dict): The validation dataset split.
            fixedNumberOfInputElements (int): The maximum sequence length.
            batch_size (int): The number of sequences per batch.
            index_to_target_word_dict (dict): The target vocabulary mapping indices to words.
            patience (int): The number of epochs to wait for improvement before reducing the learning rate.
            index_to_word_dict (dict): The source vocabulary mapping indices to words.
            timestamp (str): A unique identifier for the current training run.
            ckpt (dict, optional): A checkpoint dictionary to resume training from. Defaults to None.
            pretrained (str, optional): A flag or path for the pretrained model. Defaults to None.
        """

    # --- Setup and Initialization ---
    no_improve = 0
    best_metric = -float('inf')
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=builder.targetPAD, reduction="mean")  # Standard loss, no need to ignore padding
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    # Select the correct tokenizer based on the builder configuration (word-level vs. BPE).
    if builder.bpe==0:
        tokenizer = builder.tokenizer
    else:
        tokenizer=builder.bpe_tokenizer

    # --- Data Loading ---
    # Create the training dataset and a custom batch sampler that groups sequences of the same length.
    dataset = TranslationDataset(train_data, word_dict,target_word_dict, builder, tokenizer, fixedNumberOfInputElements)
    tqdm.write("Dataset created")
    batch_sampler = create_equal_length_batches(dataset,fixedNumberOfInputElements, batch_size)
    tqdm.write("Batch sampler created")

    # Configure the number of workers for the DataLoader based on CPU/GPU usage.
    cpu_count = multiprocessing.cpu_count()

    # Check if CUDA is available
    if torch.backends.mps.is_available() or torch.cuda.is_available():
        is_cuda = True
    else:
        is_cuda = False
    tqdm.write("Num of CPU:" f"{cpu_count }")
    tqdm.write("GPU available: "f"{ is_cuda}")
    tqdm.write("Timestamp: "f"{ timestamp}")
    if model.device == torch.device("cuda"):
        # On GPU: use more workers, but not more than available CPUs
        workers = min(8, cpu_count)
    else:
        # On CPU: use fewer workers
        workers = min(8, cpu_count)

    log_dir = os.path.join("runs", str(timestamp))
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # Initialize the DataLoader for the training set.
    train_loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=dataset.collate_equal_length_fn,
        num_workers=workers,
        pin_memory=torch.cuda.is_available()
    )

    # Create dataset
    validation_dataset = TranslationDataset(validation_data, word_dict, target_word_dict, builder, tokenizer, fixedNumberOfInputElements)
    tqdm.write("Validation Dataset created")
    # Create batches of equal length sequences
    batch_sampler_validation = create_equal_length_batches(validation_dataset, fixedNumberOfInputElements, batch_size)
    # Check number of CPUs
    tqdm.write("Batch sampler validation created")

    # Initialize the DataLoader for the validation set.
    validationLoader = DataLoader(
        validation_dataset,
        batch_sampler=batch_sampler_validation,
        collate_fn=validation_dataset.collate_equal_length_fn,
        num_workers=workers,
        pin_memory=is_cuda  # Only pin memory if using GPU
    )

    # --- Resuming from Checkpoint ---
    # If a checkpoint is provided, restore the training state.
    if ckpt is None:
        epochNumber=1
        startFineTuning = False
        global_step = 0
    else:
        epochNumber=ckpt['epoch']+1
        startFineTuning =ckpt['startFineTuning']
        best_metric = ckpt['best_metric_ChrF']
        no_improve=ckpt['no_improve']
        global_step = ckpt['global_step']

    # --- Main Training Loop ---
    # The loop continues as long as the learning rate is above the specified minimum limit.
    while optimizer.param_groups[0]['lr'] > maximumlearningRateLimit:
        model.train()
        epoch_loss = 0.0
        correct_tokens = 0
        total_tokens = 0
        tqdm.write("--------------------Training Epoch start-----------------------")
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epochNumber}")

        # --- Per-Epoch Loop (Iterating through batches) ---
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            gc.collect()
            torch.cuda.empty_cache()
            source = batch['source'].to(model.device)
            target = batch['target'].to(model.device)

            # --- On-the-fly Data Corruption ---
            # This section applies a small amount of corruption to the source data in each batch to make the model more robust.
            corrupted_ids = []
            if builder.bpe == 0:
                for sent_ids in source:
                    # Decode indices to words, apply corruption, then re-encode to indices.
                    words = [
                        index_to_word_dict.get(int(w), "<UNK>")
                        for w in sent_ids
                    ]
                    corrupted_sentences = builder.corrupt_sentence(
                        words,
                        corruption_prob=0.02,
                        times=1
                    )
                    corrupted_words = corrupted_sentences[0]
                    corrupted_idx =[]
                    for w in corrupted_words:
                        if w == "<UNK>":
                            corrupted_idx.append(builder.sourceUNK)
                        else:
                            corrupted_idx.append(word_dict.get(w, builder.sourceUNK))
                    corrupted_ids.append(corrupted_idx)

                    # Reconstruct the source tensor with the newly corrupted data.
                    source = torch.tensor(corrupted_ids, device=model.device)

            else:
                # BPE case
                source_texts = []
                for sent_ids in source:
                    clean_ids = [token_id for token_id in sent_ids.tolist() if
                                 token_id not in [builder.sourcePAD, builder.sourceEOS, builder.sourceSOS]]
                    decoded_text = builder.bpe_tokenizer.decode(clean_ids)
                    words = decoded_text.split()
                    corrupted_sent = builder.corrupt_sentence(words, corruption_prob=0.02, times=1)[0]
                    source_texts.append(" ".join(corrupted_sent))

                source_batch_encoded = builder.bpe_tokenizer.encode_batch(source_texts)
                corrupted_ids = [
                        torch.tensor([builder.sourceSOS] + enc.ids + [builder.sourceEOS], dtype=torch.long)
                        for enc in source_batch_encoded
                ]
                source = pad_sequence(corrupted_ids, batch_first=True, padding_value=builder.sourcePAD).to(model.device)

                target_texts = []
                for sent_ids in target:
                    clean_ids_tgt = [token_id for token_id in sent_ids.tolist() if
                                 token_id not in [builder.targetPAD, builder.targetEOS, builder.targetSOS]]
                    decoded_text_tgt = builder.bpe_tokenizer.decode(clean_ids_tgt)
                    words_tgt = decoded_text_tgt.split()
                    corrupted_sent_tgt= builder.corrupt_sentence(words_tgt, corruption_prob=0.02, times=1)[0]
                    target_texts.append(" ".join(corrupted_sent_tgt))
                 
                    
                target_batch_encoded = builder.bpe_tokenizer.encode_batch(target_texts)
                target_ids = [
                        torch.tensor([builder.targetSOS] + enc.ids + [builder.targetEOS], dtype=torch.long)
                        for enc in target_batch_encoded
                ]
                target = pad_sequence(target_ids, batch_first=True, padding_value=builder.targetPAD).to(model.device)

            # --- Forward and Backward Pass ---
            # Forward pass: get model predictions (logits).
            predictions, logits = model(source, target)

            # Prepare target for loss calculation: remove the initial <SOS> token.
            target_without_sos = target[:, 1:]
            eos_tensor = torch.full((target_without_sos.size(0), 1), builder.targetEOS, dtype=target_without_sos.dtype,
                                    device=target_without_sos.device)
            target_adjusted = torch.cat([target_without_sos, eos_tensor], dim=1)

            # Reshape logits and target to be compatible with CrossEntropyLoss.
            batch_size, seq_len, target_vocab_size = logits.size()
            logits_flat = logits.reshape(batch_size * seq_len, target_vocab_size)
            target_flat = target_adjusted.reshape(batch_size * seq_len)

            # Calculate the loss.
            loss = loss_fn(logits_flat, target_flat)
            writer.add_scalar('Loss/train', loss.item(), global_step=global_step)
            writer.flush()

            # Backward pass
            try:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=renormalizationLimit)
                optimizer.step()
            except:
                print("Loss.backward() skipped due to OOM, ","Batch_size: ", batch_size,"Seq_len: ", seq_len)


            # Accuracy token-level
            predicted_tokens = torch.argmax(logits, dim=-1)

            # --- Logging and Metrics for the current batch ---
            correct_tokens_batch = (predicted_tokens == target_adjusted).sum().item()
            correct_tokens += correct_tokens_batch

            total_tokens += target_adjusted.numel()
            accuracy_batch = correct_tokens_batch / target_adjusted.numel() if target_adjusted.numel() > 0 else 0.0

            writer.add_scalar('Accuracy/train', accuracy_batch, global_step=global_step)
            writer.flush()
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Accuracy': f"{accuracy_batch * 100:.2f}%",
                'Batch size': f"{source.size(0)}",
                'Sequence length': f"{source.size(1)}",
                'learning rate': f"{optimizer.param_groups[0]['lr']}",
                'fine tuning started': f"{startFineTuning}"
            })
            progress_bar.update(1)

            writer.add_scalar('GradNorm/train', grad_norm, global_step=global_step)
            writer.add_scalar("Loss/epoch_train", epoch_loss / len(train_loader), epochNumber)
            writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epochNumber)
            writer.flush()

            epoch_loss += loss.item()
            global_step += 1
            del source, target, logits, predictions, loss, logits_flat, target_flat
            torch.cuda.empty_cache()
            gc.collect()

        # --- Post-Epoch Validation and Checkpointing ---
        accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
        progress_bar.close()
        tqdm.write(
            f"Epoch {epochNumber} finished, average loss: {epoch_loss / len(train_loader)}, Accuracy: {accuracy * 100:.2f}%")

        # Run validation to get performance metrics on the validation set.
        valid_metrics = validation(
            model,
            validationLoader,
            index_to_target_word_dict,
            index_to_word=index_to_word_dict,
            beam_width=5,
            device=model.device,
            builder=builder
        )
        current_metric= valid_metrics['chrf']
        for metric_name, value in valid_metrics.items():
            writer.add_scalar(f"Validation/{metric_name}", value, epochNumber)

        # --- Early Stopping and Checkpointing Logic ---
        if epochNumber == 1:
            best_metric = current_metric
            no_improve = 0

        else:
            if current_metric > best_metric:
                best_metric = current_metric
                no_improve = 0
                torch.save({
                    'epoch': epochNumber,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'best_metric_ChrF': best_metric,
                    'startFineTuning': startFineTuning,
                    'noImprove': no_improve,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'global_step': global_step
                }, f"models/{timestamp}/best_model.pt")
                print(f"✔️  Saved best model at epoch {epochNumber} (metric={current_metric:.2f})")
            else:
                no_improve += 1
                print(f"Nessun miglioramento per {no_improve}/{patience} epoche")

            # Always save the state of the last completed epoch.

        if no_improve > patience:
            scheduler.step()
            startFineTuning=True
            print(f"PATIENCE superata → scheduler.step() invocato")
            no_improve = 0
        
        torch.save({
            'epoch': epochNumber,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_metric_ChrF': best_metric,
            'startFineTuning': startFineTuning,
            'no_improve': no_improve,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'global_step': global_step
        }, f"models/{timestamp}/last_model.pt")
        print(f" Saved current model at epoch {epochNumber} (metric={current_metric:.2f})")
        epochNumber+=1
        # Log all validation metrics to console and TensorBoard.
        tqdm.write(
            f"Validation — "
            f"BLEU: {valid_metrics['bleu']:.2f}, "
            f"CHR-F: {valid_metrics['chrf']:.2f}, "
            f"ROUGE-1: {valid_metrics['rouge1']:.4f}, "
            f"ROUGE-2: {valid_metrics['rouge2']:.4f}, "
            f"ROUGE-L: {valid_metrics['rougeL']:.4f}, "
            f"Accuracy: {valid_metrics['token_accuracy']:.2%}, "
            f"PPL: {valid_metrics['perplexity']:.2f}, "
            f"Precision: {valid_metrics['precision']:.2%}, "
            f"Recall: {valid_metrics['recall']:.2%}, "
            f"F₀.₅: {valid_metrics['f0.5']:.2%}, "
            f"F₁: {valid_metrics['f1']:.2%}, "
            f"CER: {valid_metrics['cer']:.2%}, "
            f"WER: {valid_metrics['wer']:.2%}, "
            f"SER: {valid_metrics['ser']:.2%}, "
            f"GLEU: {valid_metrics['gleu']:.2f}"
        )
