import torch
import math
import sacrebleu
from rouge_score import rouge_scorer
from tqdm import tqdm

def validation(model, validation_loader, index_to_target_word,builder, beam_width=5,  device=None):
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

    with torch.no_grad():
        for batch in tqdm(validation_loader, desc="Validation Inference"):
            # Ottieni sorgenti e target
            src = batch['source'].to(device)                 # [batch_size, src_len]
            tgt = batch['target'].to(device)                 # [batch_size, tgt_len]

            # Stampa dell'input (raw token ids)
            #print("Input ids:", src.tolist())

            # Generazione con beam search
            predictions, _, _ = beamSearch(model, src, None, beam_width, builder=builder)

            # Decodifica da id a stringhe e stampa prediction
            pred_sentences = []
            ref_sentences = []
            for pred_ids, ref_ids in zip(predictions, tgt.tolist()):
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

                # Token-level accuracy
                min_len = min(len(pred_ids), len(ref_ids))
                correct = sum((pred_ids[i] == ref_ids[i]) for i in range(min_len))
                total_correct_tokens += correct
                total_tokens += min_len

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

    #model.train()
    return {
        'bleu': bleu,
        'chrf': chrf,
        'rouge1': avg_rouge1,
        'rouge2': avg_rouge2,
        'rougeL': avg_rougeL,
        'token_accuracy': token_accuracy
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
