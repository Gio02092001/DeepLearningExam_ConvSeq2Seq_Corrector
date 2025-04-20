import gc
import math
import multiprocessing
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import sacrebleu
from rouge_score import rouge_scorer
from tqdm import tqdm

from DataLoader import TranslationDataset, create_equal_length_batches, collate_equal_length_fn



def validation(validation_data, model, tokenizer, word_dict, target_word_dict, builder,fixedNumberOfInputElements,epochNumber,writer, batch_size, validationLoader, index_to_target_word):
    #tqdm.write("Validation started.")
    model.eval()
    #loss_fn = torch.nn.CrossEntropyLoss()  # Standard loss, no need to ignore padding
    global_step = 0

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
        tqdm.write("--------------------Validation Epoch start-----------------------")
        time.sleep(0.1)
        progress_bar = tqdm(validationLoader, desc=f"Epoch {epochNumber} Validation")

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
            #predictedSequence = []
            log_probs_collected = []
            all_predictions = []
            all_references = []
            predicted_ids, total_nll, total_tokens = beamSearch(model, source,progress_bar, beam_width=5, builder=builder)
            predictedSequence = torch.tensor(predicted_ids, dtype=torch.long, device=model.device)
            """for step in range(source.shape[1]):  # You should define this limit!
                predictions, logits = model(source, targetInput)

                log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)

                correct_log_probs = log_probs.gather(1, target[:, step].unsqueeze(1)).squeeze()

                total_nll_batch += -correct_log_probs.sum().item()
                total_tokens_batch += target.size(0)  # aggiungi batch_size ogni step

                next_tokens = torch.argmax(log_probs, dim=1, keepdim=True)
                targetInput = torch.cat([targetInput, next_tokens], dim=1)

                predictedSequence.append(next_tokens)"""

            total_nll += total_nll_batch
            total_tokens += total_tokens_batch

            # After the loop, stack the sequence
            #predictedSequence = torch.cat(predictedSequence, dim=-1)  # Shape: (batch_size, max_output_length)
            #tqdm.write(predictedSequence)
            # Stack all log probabilities
            if total_tokens > 0:
                average_nll = total_nll / total_tokens
                perplexity = math.exp(average_nll)
            else:
                average_nll = float('nan')
                perplexity = float('nan')

            epoch_loss += average_nll

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
                #tqdm.write("Prediction: ", sentence)

                pred_sentences.append(" ".join(sentence))

            ref_sentences= []
            for sequence in target.tolist():
                sentence = []
                for token_id in sequence:
                    word = index_to_target_word[token_id]
                    if  word in ["<pad>", "<sos>", "<eos>"]:
                        continue
                    sentence.append(word)
                #tqdm.write("Reference: ", sentence)
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
            writer.flush()

            #tqdm.write(
            #    f"VALIDATION Batch {batch_idx}, Loss: {average_nll}, Perplexity: {perplexity}, Accuracy: {token_accuracy*100} , Batch size: {source.size(0)}, Sequence length: {source.size(1)}")
            # Backward pass

            #epoch_loss += loss.item()
            epoch_perplexity +=perplexity
            writer.add_scalar('Perplexity/validation_epoch', epoch_perplexity, global_step=counter)
            writer.flush()

            global_step += 1

            del source, target
            torch.cuda.empty_cache()
            gc.collect()
        # Dopo aver completato il ciclo sui batch:
    progress_bar.close()
    bleu = sacrebleu.corpus_bleu(all_predictions, [all_references])
    tqdm.write(f"BLEU Score: {bleu.score}")

    chrf = sacrebleu.corpus_chrf(all_predictions, [all_references])
    tqdm.write(f"CHRF Score: {chrf.score}")

    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1, rouge2, rougeL = [], [], []

    for pred, ref in zip(all_predictions, all_references):
        scores = scorer.score(ref, pred)
        rouge1.append(scores['rouge1'].fmeasure)
        rouge2.append(scores['rouge2'].fmeasure)
        rougeL.append(scores['rougeL'].fmeasure)

    tqdm.write(f"ROUGE-1: {sum(rouge1) / len(rouge1):.4f}")
    tqdm.write(f"ROUGE-2: {sum(rouge2) / len(rouge2):.4f}")
    tqdm.write(f"ROUGE-L: {sum(rougeL) / len(rougeL):.4f}")

    # Log metrics to TensorBoard
    writer.add_scalar('BLEU/validation_epoch', bleu.score, global_step=counter)
    writer.add_scalar('CHRF/validation_epoch', chrf.score, global_step=counter)
    writer.add_scalar('ROUGE1/validation_epoch', sum(rouge1) / len(rouge1), global_step=counter)
    writer.add_scalar('ROUGE2/validation_epoch', sum(rouge2) / len(rouge2), global_step=counter)
    writer.add_scalar('ROUGEL/validation_epoch', sum(rougeL) / len(rougeL), global_step=counter)
    writer.flush()
    epoch_perplexity=epoch_perplexity/ len(validationLoader)

    tqdm.write(f"Epoch {epochNumber} finished, average loss: {epoch_loss / len(validationLoader)}")
    tqdm.write(f"Epoch perplexity: {epoch_perplexity}")
    return epoch_perplexity

def beamSearch(model, source,progress_bar, beam_width, builder, max_output_length=100):
    """
    Esegue Beam Search su un batch di sequenze sorgenti.

    Args:
        model: Il tuo modello Seq2Seq.
        source: Tensor di input (batch_size, source_len).
        beam_width: Numero di ipotesi da mantenere per frase.
        builder: Oggetto con targetSOS.
        max_output_length: Lunghezza massima della sequenza generata.

    Returns:
        final_sequences: Lista di liste di token_id predetti.
    """
    batch_size = source.shape[0]
    device = model.device

    # Inizializza le sequenze con solo SOS token
    sequences = torch.full((batch_size, beam_width, 1), builder.targetSOS, dtype=torch.long, device=device)
    scores = torch.zeros((batch_size, beam_width), dtype=torch.float, device=device)

    # Variabili per NLL e tokens
    total_nll = 0.0
    total_tokens = 0

    for step in range(max_output_length):
        num_candidates = sequences.size(1)  # beam_width

        # Preparazione input per il modello
        input_seq = sequences.view(batch_size * num_candidates, -1)
        source_repeated = source.unsqueeze(1).repeat(1, num_candidates, 1).view(batch_size * num_candidates, -1)

        # Model forward
        _, logits = model(source_repeated, input_seq)
        log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)

        # Aggiorna i punteggi sommando log_probs
        expanded_scores = scores.view(-1, 1) + log_probs  # [batch_size * beam_width, vocab_size]

        # Seleziona le migliori beam_width ipotesi per frase
        expanded_scores = expanded_scores.view(batch_size, -1)
        top_scores, top_indices = torch.topk(expanded_scores, beam_width, dim=-1)

        # Calcola NLL per il passo corrente
        nll = -top_scores.sum().item()  # Somma dei log dei punteggi
        total_nll += nll

        # Aggiungi il numero di token per il passo corrente
        total_tokens += batch_size * beam_width  # Ogni candidato ha un token

        # Ricostruisci le nuove sequenze
        beam_indices = top_indices // log_probs.shape[-1]
        token_indices = top_indices % log_probs.shape[-1]

        new_sequences = []
        for i in range(batch_size):
            temp = []
            for b in range(beam_width):
                old_seq = sequences[i, beam_indices[i, b]].tolist()
                token = token_indices[i, b].item()
                temp.append(old_seq + [token])
            new_sequences.append(temp)

        sequences = torch.tensor(new_sequences, dtype=torch.long, device=device)
        scores = top_scores

    # Normalizza per lunghezza e scegli la sequenza migliore
    normalized_scores = scores / sequences.size(-1)
    best_indices = normalized_scores.argmax(dim=1)

    final_sequences = []
    for i in range(batch_size):
        best_seq = sequences[i, best_indices[i]].tolist()
        final_sequences.append(best_seq)

    # Dopo la generazione, calcola la perdita media
    average_nll = total_nll / total_tokens if total_tokens > 0 else 0.0
    progress_bar.set_postfix({
        'Total NLL': f'{total_nll:.4f}',
        'Total Tokens': total_tokens,
        'Avg NLL/Token': f'{average_nll:.4f}'
    })
    progress_bar.update(1)

    return final_sequences, total_nll, total_tokens
