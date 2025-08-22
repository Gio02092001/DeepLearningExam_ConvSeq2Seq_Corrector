import jiwer
import torch
import math
import sacrebleu
from rouge_score import rouge_scorer
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from nltk.translate.gleu_score import sentence_gleu
import editdistance





def validation(model, validation_loader, index_to_target_word, index_to_word, builder, beam_width=5,  device=None):
    """
    Esegue la validazione in modalità inferenza (senza teacher forcing).

    Args:
        model: modello Seq2Seq già caricato
        validation_loader: DataLoader per il set di validazione
        index_to_target_word: mappatura da id a parola per decodifica
        beam_width: larghezza del beam per la generazione
        device: dispositivo (cpu o cuda)

    Returns:
        Dictionary con BLEU, CHRF, ROUGE-1/2/L e accuracy token-level.
    """
    tqdm.write("--------------------Validation start-----------------------")
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    all_hypotheses = []
    all_references = []
    total_correct_tokens = 0
    total_tokens = 0
    total_loss = 0.0
    total_ppl_tokens = 0

    total_fp = 0
    total_sentences = 0
    total_sentence_errors = 0
    total_character_errors = 0
    total_characters = 0
    total_word_errors = 0
    total_words = 0
    all_preds_bin = []
    all_targets_bin = []
    gleu_scores = []

    pad_token_id = builder.targetPAD  # Assicurati che sia corretto
    loss_val = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction="sum")

    with torch.no_grad():
        for batch in tqdm(validation_loader, desc="Validation Inference"):
            # Ottieni sorgenti e target
            src = batch['source'].to(device)                 # [batch_size, src_len]
            tgt = batch['target'].to(device)                 # [batch_size, tgt_len]

            # + passata teacher-forcing per somma della cross‐entropy
            _, logits_tf = model(src, tgt)
            # + loss sommata su tutti i token non-pad
            loss_tf = loss_val(
                logits_tf.view(-1, logits_tf.size(-1)),
                tgt.view(-1)
            ).item()
            # + conto token non-pad
            # nonpad = (tgt != pad_token_id).sum().item()
            nonpad = (tgt != pad_token_id).sum().item()
            total_loss += loss_tf
            total_ppl_tokens += nonpad
            # Stampa dell'input (raw token ids)
            #print("Input ids:", src.tolist())

            # Generazione con beam search
            predictions, _, _ = beamSearch(model, src, None, beam_width, builder=builder)

            # Decodifica da id a stringhe e stampa prediction
            pred_sentences = []
            ref_sentences = []
            for pred_ids, ref_ids, inp_ids in zip(predictions, tgt.tolist(), src.tolist()):

                # Decodifica predizione
                words = [index_to_target_word[i] for i in pred_ids
                         if index_to_target_word[i] not in ['<pad>','<sos>','<eos>']]
                sentence = " ".join(words)
                #print("Prediction:", sentence)
                pred_sentences.append(sentence)

                # Decodifica riferimento
                ref_words = [index_to_target_word[i] for i in ref_ids
                             if index_to_target_word[i] not in ['<pad>','<sos>','<eos>']]
                ref_sentence = " ".join(ref_words)
                ref_sentences.append(ref_sentence)

                inp_words = [index_to_word[i] for i in inp_ids
                             if index_to_word[i] not in ['<pad>', '<sos>', '<eos>']]


                # Token-level accuracy
                min_len = min(len(pred_ids), len(ref_ids))
                correct = sum((pred_ids[i] == ref_ids[i]) for i in range(min_len))
                total_correct_tokens += correct
                total_tokens += min_len

                # ✅ GLEU
                gleu_scores.append(sentence_gleu([ref_words], words))

                # ✅ CER
                total_character_errors += editdistance.eval(''.join(words), ''.join(ref_words))
                total_characters += len(''.join(ref_words))

                # ✅ WER
                total_word_errors += editdistance.eval(words, ref_words)
                total_words += len(ref_words)

                # ✅ SER
                total_sentence_errors += int(words != ref_words)
                total_sentences += 1

                # ✅ Precision/Recall binario
                pred_change = int(words != inp_words)
                ref_change = int(ref_words != inp_words)
                all_preds_bin.append(pred_change)
                all_targets_bin.append(ref_change)

                if pred_change and not ref_change:
                    total_fp += 1


            all_hypotheses.extend(pred_sentences)
            all_references.extend(ref_sentences)

    # Calcolo metriche corpus-level
    bleu = sacrebleu.corpus_bleu(all_hypotheses, [all_references]).score
    chrf = sacrebleu.corpus_chrf(all_hypotheses, [all_references]).score
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    rouge1, rouge2, rougeL = [], [], []
    for ref, hyp in zip(all_references, all_hypotheses):
        scores = scorer.score(ref, hyp)
        rouge1.append(scores['rouge1'].fmeasure)
        rouge2.append(scores['rouge2'].fmeasure)
        rougeL.append(scores['rougeL'].fmeasure)

    avg_rouge1 = sum(rouge1)/len(rouge1) if rouge1 else 0.0
    avg_rouge2 = sum(rouge2)/len(rouge2) if rouge2 else 0.0
    avg_rougeL = sum(rougeL)/len(rougeL) if rougeL else 0.0
    token_accuracy = total_correct_tokens/total_tokens if total_tokens>0 else 0.0

    ppl = math.exp(total_loss / total_ppl_tokens)

    precision = precision_score(all_targets_bin, all_preds_bin, zero_division=0)
    recall = recall_score(all_targets_bin, all_preds_bin, zero_division=0)
    f1 = f1_score(all_targets_bin, all_preds_bin, zero_division=0)
    f05 = (1.25 * precision * recall) / (0.25 * precision + recall + 1e-8) if (precision + recall) > 0 else 0.0
    accuracy_bin = sum(int(p == r) for p, r in zip(all_preds_bin, all_targets_bin)) / len(all_preds_bin)
    false_positive_rate = total_fp / (
                total_fp + sum(int(not r and not p) for p, r in zip(all_preds_bin, all_targets_bin)) + 1e-8)
    cer = jiwer.cer(ref_sentences, pred_sentences)
    wer = jiwer.wer(ref_sentences, pred_sentences)
    ser = total_sentence_errors / total_sentences if total_sentences > 0 else 0.0
    gleu = sum(gleu_scores) / len(gleu_scores) if gleu_scores else 0.0

   # Check these metrics non mi convincono

    #model.train()
    return {
        'bleu': bleu,
        'chrf': chrf,
        'rouge1': avg_rouge1,
        'rouge2': avg_rouge2,
        'rougeL': avg_rougeL,
        'token_accuracy': token_accuracy,
        'perplexity': ppl,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f0.5': f05,
        'binary_accuracy': accuracy_bin,
        'false_positive_rate': false_positive_rate,
        'cer': cer,
        'wer': wer,
        'ser': ser,
        'gleu': gleu
    }


def beamSearch(model, source, progress_bar, beam_width, builder, max_output_length=100):
    # Come definito precedentemente, assicura che restituisca solo sequenze.
    batch_size = source.size(0)
    device = source.device
    sequences = torch.full((batch_size, beam_width, 1), builder.targetSOS, dtype=torch.long, device=device)
    scores = torch.zeros((batch_size, beam_width), dtype=torch.float, device=device)

    for _ in range(max_output_length):
        num_candidates = sequences.size(1)
        input_seq = sequences.view(batch_size * num_candidates, -1)
        source_rep = source.unsqueeze(1).repeat(1, num_candidates, 1).view(batch_size * num_candidates, -1)
        _, logits = model(source_rep, input_seq)
        log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)

        expanded_scores = scores.view(-1,1) + log_probs
        expanded_scores = expanded_scores.view(batch_size, -1)
        top_scores, top_indices = torch.topk(expanded_scores, beam_width, dim=1)

        beam_idx = top_indices // log_probs.size(-1)
        token_idx = top_indices % log_probs.size(-1)

        # Ricostruzione sequenze
        new_seqs = []
        for i in range(batch_size):
            new_seqs.append([
                sequences[i, beam_idx[i,b]].tolist() + [token_idx[i,b].item()]
                for b in range(beam_width)
            ])
        sequences = torch.tensor(new_seqs, device=device)
        scores = top_scores

    # Selezione migliore ipotesi normalizzata per lunghezza
    lengths = sequences.size(-1)
    norm_scores = scores / lengths
    best_idx = norm_scores.argmax(dim=1)
    final = [ sequences[i, best_idx[i]].tolist() for i in range(batch_size) ]

    return final, None, None
