import gc
import math
import multiprocessing
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import sacrebleu
from rouge_score import rouge_scorer
from DataLoader import TranslationDataset, create_equal_length_batches, collate_equal_length_fn

PER_BEAM_SEARCH = """POI QUANDO QUESTO FUNZIONA CAMBIA ARGMAX PER BEAM SEARCH"""


def validation(validation_data, model, tokenizer, word_dict, target_word_dict, builder,fixedNumberOfInputElements,epochNumber, batch_size=64):
    print("Validation started.")
    model.eval()
    #loss_fn = torch.nn.CrossEntropyLoss()  # Standard loss, no need to ignore padding
    global_step = 0
    index_to_target_word = {index: word for word, index in target_word_dict.items()}
    # Create dataset
    dataset = TranslationDataset(validation_data, word_dict, target_word_dict, builder, tokenizer)
    print("Dataset created")
    # Create batches of equal length sequences
    batch_sampler = create_equal_length_batches(dataset, fixedNumberOfInputElements, batch_size)
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
    total_nll=0

    total_edit_distance = 0
    num_sequences = 0
    with torch.no_grad():
        # Iterate through batches
        for batch_idx, batch in enumerate(validationLoader):
            total_nll_batch = 0
            total_tokens_batch = 0
            gc.collect()
            torch.cuda.empty_cache()

            source = batch['source'].to(model.device)
            target = batch['target'].to(model.device)  # If you want to compare later
            sourceBatchSize = source.shape[0]

            # Start with only SOS tokens
            targetInput = torch.full((sourceBatchSize, 1), builder.targetSOS, dtype=torch.long, device=model.device)

            # Placeholder to collect predictions
            predictedSequence = []
            log_probs_collected = []
            all_predictions = []
            all_references = []

            for step in range(source.shape[1]):  # You should define this limit!
                predictions, logits = model(source, targetInput)

                log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)

                correct_log_probs = log_probs.gather(1, target[:, step].unsqueeze(1)).squeeze()

                total_nll_batch += -correct_log_probs.sum().item()
                total_tokens_batch += target.size(0)  # aggiungi batch_size ogni step

                next_tokens = torch.argmax(log_probs, dim=1, keepdim=True)
                targetInput = torch.cat([targetInput, next_tokens], dim=1)

                predictedSequence.append(next_tokens)

            total_nll += total_nll_batch
            total_tokens += total_tokens_batch

            # After the loop, stack the sequence
            predictedSequence = torch.cat(predictedSequence, dim=-1)  # Shape: (batch_size, max_output_length)
            #print(predictedSequence)
            # Stack all log probabilities
            average_nll = total_nll / total_tokens
            perplexity = math.exp(average_nll)

            # Token-Level Accuracy
            min_len = min(predictedSequence.shape[1], target.shape[1])
            correct_tokens += (predictedSequence[:, :min_len] == target[:, :min_len]).sum().item()



            # Converti token in stringa
            pred_sentences = []
            for sequence in predictedSequence.tolist():
                sentence = []
                for token_id in sequence:
                    word = index_to_target_word[token_id]
                    if  word in ["<pad>", "<sos>", "<eos>"]:
                        continue
                    sentence.append(word)
                pred_sentences.append(" ".join(sentence))

            ref_sentences= []
            for sequence in target.tolist():
                sentence = []
                for token_id in sequence:
                    word = index_to_target_word[token_id]
                    if  word in ["<pad>", "<sos>", "<eos>"]:
                        continue
                    sentence.append(word)
                ref_sentences.append(" ".join(sentence))

            all_predictions.extend(pred_sentences)
            all_references.extend(ref_sentences)



            # Calculate loss
            # Reshape logits to match CrossEntropyLoss expectations
            # Assuming logits has shape [batch_size, seq_len, vocab_size]
            #batch_size, seq_len, target_vocab_size = logits.size()
            #logits_flat = logits.reshape(batch_size * seq_len, target_vocab_size)
            #target_flat = target.reshape(batch_size * seq_len)

            #loss = loss_fn(logits_flat, target_flat)

            #perplexity = torch.exp(total_nll)
            token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
            writer.add_scalar('Loss/validation_batch', average_nll, global_step=global_step)
            writer.add_scalar('accuracy/validation_batch', token_accuracy, global_step=global_step)

            print(
                f"VALIDATION Batch {batch_idx}, Loss: {average_nll}, Perplexity: {perplexity}, Accuracy: {token_accuracy*100} , Batch size: {source.size(0)}, Sequence length: {source.size(1)}")
            # Backward pass

            #epoch_loss += loss.item()
            epoch_perplexity +=perplexity
            writer.add_scalar('Perplexity/validation_epoch', epoch_perplexity, global_step=counter)

            global_step += 1

            del source, target, logits, predictions
            torch.cuda.empty_cache()
            gc.collect()
        # Dopo aver completato il ciclo sui batch:
    bleu = sacrebleu.corpus_bleu(all_predictions, [all_references])
    print(f"BLEU Score: {bleu.score}")

    chrf = sacrebleu.corpus_chrf(all_predictions, [all_references])
    print(f"CHRF Score: {chrf.score}")

    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1, rouge2, rougeL = [], [], []

    for pred, ref in zip(all_predictions, all_references):
        scores = scorer.score(ref, pred)
        rouge1.append(scores['rouge1'].fmeasure)
        rouge2.append(scores['rouge2'].fmeasure)
        rougeL.append(scores['rougeL'].fmeasure)

    print(f"ROUGE-1: {sum(rouge1) / len(rouge1):.4f}")
    print(f"ROUGE-2: {sum(rouge2) / len(rouge2):.4f}")
    print(f"ROUGE-L: {sum(rougeL) / len(rougeL):.4f}")

    # Log metrics to TensorBoard
    writer.add_scalar('BLEU/validation_epoch', bleu.score, global_step=counter)
    writer.add_scalar('CHRF/validation_epoch', chrf.score, global_step=counter)
    writer.add_scalar('ROUGE1/validation_epoch', sum(rouge1) / len(rouge1), global_step=counter)
    writer.add_scalar('ROUGE2/validation_epoch', sum(rouge2) / len(rouge2), global_step=counter)
    writer.add_scalar('ROUGEL/validation_epoch', sum(rougeL) / len(rougeL), global_step=counter)
    epoch_perplexity=epoch_perplexity/ len(validationLoader)

    print(f"Epoch {epochNumber} finished, average loss: {epoch_loss / len(validationLoader)}")
    print(f"Epoch perplexity: {epoch_perplexity}")
    return epoch_perplexity

