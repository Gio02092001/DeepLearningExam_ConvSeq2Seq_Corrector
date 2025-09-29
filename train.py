import gc
import multiprocessing
import os
import time
from datetime import datetime
import random
import shutil
import os

import torch
from nltk import TweetTokenizer
from sacremoses import MosesTokenizer
from tensorflow import timestamp
from torch.utils.data import DataLoader
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

from DataLoader import TranslationDataset, create_equal_length_batches

from validation import validation



def train(model, optimizer, scheduler, train_data, builder, word_dict, renormalizationLimit, maximumlearningRateLimit,
          target_word_dict,validation_data,fixedNumberOfInputElements, batch_size, index_to_target_word_dict, patience, index_to_word_dict,timestamp, ckpt=None, pretrained=None):
    model.train()

        
    patience = patience
    no_improve = 0
    best_metric = -float('inf')
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")  # Standard loss, no need to ignore padding
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    if builder.bpe==0:
        tokenizer = builder.tokenizer
    else:
        tokenizer=builder.bpe_tokenizer


    # Create dataset
    dataset = TranslationDataset(train_data, word_dict,target_word_dict, builder, tokenizer)
    tqdm.write("Dataset created")
    # Create batches of equal length sequences
    batch_sampler = create_equal_length_batches(dataset,fixedNumberOfInputElements, batch_size)
    # Check number of CPUs
    tqdm.write("Batch sampler created")
    cpu_count = multiprocessing.cpu_count()

    # Check if CUDA is available
    if torch.backends.mps.is_available() or torch.cuda.is_available():
        is_cuda = True
    else:
        is_cuda = False
    tqdm.write("Num of CPU:" f"{cpu_count }")
    tqdm.write("GPU available: "f"{ is_cuda}")
    tqdm.write("Timestamp: "f"{ timestamp}")

    # Convert it to a string

    # Decide number of workers
    # Decide number of workers
    if model.device == torch.device("cuda"):
        # On GPU: use more workers, but not more than available CPUs
        workers = min(8, cpu_count)

    else:
        # On CPU: use fewer workers
        workers = min(8, cpu_count)

    log_dir = os.path.join("runs", str(timestamp))
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # DataLoader
    train_loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=dataset.collate_equal_length_fn,
        num_workers=workers,
        pin_memory=torch.cuda.is_available()  # Only pin memory if using GPU
    )

    # Create dataset
    validation_dataset = TranslationDataset(validation_data, word_dict, target_word_dict, builder, tokenizer)
    tqdm.write("Validation Dataset created")
    # Create batches of equal length sequences
    batch_sampler_validation = create_equal_length_batches(validation_dataset, fixedNumberOfInputElements, batch_size)
    # Check number of CPUs
    tqdm.write("Batch sampler validation created")

    # DataLoader
    validationLoader = DataLoader(
        validation_dataset,
        batch_sampler=batch_sampler_validation,
        collate_fn=validation_dataset.collate_equal_length_fn,
        num_workers=workers,
        pin_memory=is_cuda  # Only pin memory if using GPU
    )
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


    while optimizer.param_groups[0]['lr'] > maximumlearningRateLimit:
        #validation_output= validation(validation_data, model, tokenizer, word_dict, target_word_dict, builder, fixedNumberOfInputElements, epochNumber, writer, batch_size, validationLoader, index_to_target_word_dict)
        #tqdm.write("FineTuning started: ", startFineTuning)
        #tqdm.write(f"Current learning rate: {optimizer.param_groups[0]['lr']}")

        epoch_loss = 0.0
        correct_tokens = 0  # Inizializza il contatore dei token corretti
        total_tokens = 0
        #time.sleep(0.1)
        tqdm.write("--------------------Training Epoch start-----------------------")
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epochNumber}")

        # Iterate through batches
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            gc.collect()
            torch.cuda.empty_cache()

            source = batch['source'].to(model.device)
            target = batch['target'].to(model.device)

            # 1) Estrai source_ids e portali su CPU
            source_begin=source

            corrupted_ids = []
            if builder.bpe == 0:
                for sent_ids in source:
                    # 2) idx → parola

                    words = [
                        index_to_word_dict.get(int(w), "<UNK>")
                        for w in sent_ids
                    ]
                    # 3) applica corrupt_sentence: ottieni una lista di frasi (times versioni)
                    corrupted_sentences = builder.corrupt_sentence(
                        words,
                        corruption_prob=0.02,
                        times=1
                    )

                    # 4) prendi la prima (o randomly una delle volte, se vuoi variarlo)
                    corrupted_words = corrupted_sentences[0].split()

                    # 5) parola → idx, con fallback su unk
                    corrupted_idx =[]
                    for w in corrupted_words:
                        if w == "<UNK>":
                            corrupted_idx.append(builder.sourceUNK)
                        else:
                            corrupted_idx.append(word_dict.get(w, builder.sourceUNK))

                    corrupted_ids.append(corrupted_idx)
                    # 7) ricostruisci il tensore e rimandalo su GPU
                    try:
                        source = torch.tensor(corrupted_ids, device=model.device)
                    except:
                        print(corrupted_words)
                        print("Begin: ", source_begin)
            else:
                source_texts = []
                for sent_ids in source:
                    words = [index_to_word_dict.get(int(w), "<unk>") for w in sent_ids]
                    sentence = " ".join(words)
                    corrupted_sent = builder.corrupt_sentence(
                        words,
                        corruption_prob=0.02,
                        times=1
                    )[0]  # prendi la prima
                    source_texts.append(corrupted_sent)

                corrupted_batch = builder.bpe_tokenizer.encode_batch(source_texts)
                corrupted_ids = [torch.tensor([builder.sourceSOS] + enc.ids + [builder.sourceEOS], dtype=torch.long)
                  for enc in corrupted_batch]
                source = pad_sequence(corrupted_ids, batch_first=True, padding_value=builder.sourcePAD).to(model.device)
                # target: idx → testo → encode con BPE
                target_texts = []
                for sent_ids in target:
                    words = [index_to_target_word_dict.get(int(w), "<unk>") for w in sent_ids]
                    sentence = " ".join(words)
                    target_texts.append(sentence)

                target_batch_encoded = builder.bpe_tokenizer.encode_batch(target_texts)
                target_ids = [torch.tensor([builder.targetSOS] + enc.ids + [builder.targetEOS], dtype=torch.long)
                                for enc in target_batch_encoded]
                target = pad_sequence(target_ids, batch_first=True, padding_value=builder.targetPAD).to(model.device)





            # Forward pass with batch
            predictions, logits = model(source, target)

            target_without_sos = target[:, 1:]  # Rimuove il primo token (SOS)
            eos_tensor = torch.full((target_without_sos.size(0), 1), builder.targetEOS, dtype=target_without_sos.dtype,
                                    device=target_without_sos.device)
            target_adjusted = torch.cat([target_without_sos, eos_tensor], dim=1)  # Aggiungi EOS

            # Loss: reshape logits per CrossEntropyLoss
            batch_size, seq_len, target_vocab_size = logits.size()
            logits_flat = logits.reshape(batch_size * seq_len, target_vocab_size)
            target_flat = target_adjusted.reshape(batch_size * seq_len)

            # Calcola la loss
            loss = loss_fn(logits_flat, target_flat)
            writer.add_scalar('Loss/train', loss.item(), global_step=global_step)
            writer.flush()

            # Accuracy token-level
            predicted_tokens = torch.argmax(logits, dim=-1)

            # Conta token corretti
            correct_tokens_batch = (predicted_tokens == target_adjusted).sum().item()
            correct_tokens += correct_tokens_batch  # aggiorna conteggio

            total_tokens += target_adjusted.numel()  # Conta tutti i token nel target modificato
            accuracy_batch = correct_tokens_batch / target_adjusted.numel() if target_adjusted.numel() > 0 else 0.0

            #if accuracy_batch > 0.5:
                #print(accuracy_batch)
                #for pred, targ in zip(predicted_tokens, target_adjusted):
                    #print("prediction: ", pred)
                    #print("target: ", targ)
                    #print("------------------------")
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
            #tqdm.write(
                #f"Batch {batch_idx}, Loss: {loss.item()}, Batch size: {source.size(0)}, Accuracy: {accuracy_batch}, Sequence length: {source.size(1)}")
            # Backward pass
            loss.backward()
            grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=renormalizationLimit)
            writer.add_scalar('GradNorm/train', grad_norm, global_step=global_step)
            writer.add_scalar("Loss/epoch_train", epoch_loss / len(train_loader), epochNumber)
            writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epochNumber)
            writer.flush()

            optimizer.step()
            epoch_loss += loss.item()
            global_step += 1
            del source, target, logits, predictions, loss, logits_flat, target_flat
            torch.cuda.empty_cache()
            gc.collect()



        accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
        progress_bar.close()
        tqdm.write(
            f"Epoch {epochNumber} finished, average loss: {epoch_loss / len(train_loader)}, Accuracy: {accuracy * 100:.2f}%")



        # ... dentro al loop di epoca, subito dopo aver chiuso progress_bar ...
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
        for metric_name, value in valid_metrics.items():
            writer.add_scalar(f"Validation/{metric_name}", value, epochNumber)

        # gestione della patience
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
        # solo quando no_improve >= PATIENCE, chiamo scheduler.step()
        if no_improve > patience:
            # scheduler.step() ridurrà lr di 10× (fino al tuo min_lr 1e-4)
            scheduler.step()
            startFineTuning=True
            print(f"PATIENCE superata → scheduler.step() invocato")
            no_improve = 0  # resetta contatore per misurare le prossime PATIENCE

        epochNumber+=1


def tokenizeSentence(input_sentence):
    mt = MosesTokenizer('en')
    return mt.tokenize(input_sentence)

