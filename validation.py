import jiwer
import torch
import math
import sacrebleu
from rouge_score import rouge_scorer
from tqdm import tqdm
from nltk.translate.gleu_score import sentence_gleu
import editdistance
from sklearn.metrics import precision_recall_fscore_support
import warnings
import gc
import difflib

def trim_and_filter(ids, pad_id, sos_id, eos_id):
    """Remove PAD/SOS tokens and truncate at first EOS"""
    out = []
    for t in ids:
        if t == pad_id or t == sos_id:
            continue
        if t == eos_id:
            break
        out.append(t)
    return out

def lcs_match_count(a, b):
    """Count matching tokens using Longest Common Subsequence"""
    sm = difflib.SequenceMatcher(None, a, b)
    return sum(m.size for m in sm.get_matching_blocks())


def validation(model, validation_loader, index_to_target_word, index_to_word, builder, beam_width=5,  device=None):
    """
    Performs a full validation run on the model using inference mode (no teacher forcing for generation).

    This function calculates a comprehensive set of metrics, including BLEU, CHR-F, ROUGE,
    perplexity, token-level accuracy, precision/recall/F1, CER, WER, SER, and GLEU.

    Args:
        model (torch.nn.Module): The sequence-to-sequence model to evaluate.
        validation_loader (DataLoader): The DataLoader for the validation dataset.
        index_to_target_word (dict): A dictionary mapping target vocabulary indices to words/tokens.
        index_to_word (dict): A dictionary mapping source vocabulary indices to words/tokens.
        builder (BuildDictionary_Map): The data builder instance, used for special token IDs and BPE info.
        beam_width (int): The beam width for beam search decoding.
        device (torch.device, optional): The device (e.g., 'cuda' or 'cpu') to run evaluation on.

    Returns:
        dict: A dictionary containing the calculated evaluation metrics.
    """
    # Suppress UserWarning from sklearn when dealing with labels that are not present in predictions.
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    tqdm.write("--------------------Validation start-----------------------")
    # Set the model to evaluation mode. This disables layers like dropout.
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    all_hypotheses = []
    all_references = []
    total_correct_tokens = 0
    total_tokens = 0
    total_loss = 0.0
    total_ppl_tokens = 0
    gleu_scores = []

    pad_token_id = builder.targetPAD
    # Define the loss function for perplexity, ignoring the padding token index and summing the loss over the batch.
    loss_val = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction="sum")

    # Disable gradient calculations for validation
    with torch.no_grad():
        for batch in tqdm(validation_loader, desc="Validation Inference"):
            # Move source and target tensors to the specified device.
            src = batch['source'].to(device)
            tgt = batch['target'].to(device)
            
            # --- Perplexity Calculation (using teacher forcing) ---
            # Pass both source and target to the model to get logits for the loss calculation.
            _, logits_tf = model(src, tgt)
            # Calculate the cross-entropy loss sum over all non-padded tokens in the batch.
            loss_tf = loss_val(
                logits_tf.view(-1, logits_tf.size(-1)),
                tgt.view(-1)
            ).item()
            # Count the number of non-padded tokens for normalization.
            nonpad = (tgt != pad_token_id).sum().item()
            total_loss += loss_tf
            total_ppl_tokens += nonpad

            # --- Sentence Generation (using beam search) ---
            predictions, _, _ = beamSearch(model, src, None, beam_width, builder=builder)
            
            # --- Decode and Calculate Batch Metrics ---
            pred_sentences = []
            ref_sentences = []
            gleu_scores = []

            # ADD THESE THREE LINES:
            pr_matches = 0
            pr_ref_len = 0
            pr_hyp_len = 0
            for pred_ids, ref_ids, inp_ids in zip(predictions, tgt.tolist(), src.tolist()):
                if builder.bpe == 0:
                    # --- Word-Level Decoding ---
                    words = [index_to_target_word[i] for i in pred_ids
                             if index_to_target_word[i] not in ['<pad>', '<sos>', '<eos>']]
                    sentence = " ".join(words)

                    # Decode reference sentence.
                    ref_words = [index_to_target_word[i] for i in ref_ids
                                 if index_to_target_word[i] not in ['<pad>', '<sos>', '<eos>']]
                    ref_sentence = " ".join(ref_words)

                else:
                    # --- BPE-Level Decoding ---
                    # Clean IDs: remove PAD/SOS and truncate at first EOS
                    pred_ids_clean = trim_and_filter(pred_ids, builder.targetPAD, builder.targetSOS, builder.targetEOS)
                    ref_ids_clean = trim_and_filter(ref_ids, builder.targetPAD, builder.targetSOS, builder.targetEOS)
                    
                    # Decode BPE IDs to text strings (using YOUR existing tokenizer)
                    sentence = builder.bpe_tokenizer.decode(pred_ids_clean).strip()
                    ref_sentence = builder.bpe_tokenizer.decode(ref_ids_clean).strip()
                    
                    # Split into words using simple split (same as you had)
                    words = sentence.split()
                    ref_words = ref_sentence.split()
                
                # Append the decoded string sentences to the corpus lists.
                pred_sentences.append(sentence)
                ref_sentences.append(ref_sentence)
                
                if builder.bpe == 1:
                    # --- Token-level Accuracy via LCS alignment on IDs ---
                    matches = lcs_match_count(ref_ids_clean, pred_ids_clean)
                    total_correct_tokens += matches
                    total_tokens += len(ref_ids_clean)

                    # Accumulate word-level matches for P/R/F1
                    pr_matches += lcs_match_count(ref_words, words)
                    pr_ref_len += len(ref_words)
                    pr_hyp_len += len(words)

                # Calculate GLEU score for the current sentence.
                gleu_scores.append(sentence_gleu([ref_words], words))
                # --- Calculate Token-level Accuracy for the current sentence ---
                min_len = min(len(pred_ids), len(ref_ids))
                correct = sum((pred_ids[i] == ref_ids[i]) for i in range(min_len))
                total_correct_tokens += correct
                total_tokens += min_len

                # Calculate GLEU score for the current sentence.
                gleu_scores.append(sentence_gleu([ref_words], words))


            # Add the processed sentences from this batch to the overall lists.
            all_hypotheses.extend(pred_sentences)
            all_references.extend(ref_sentences)
            # Clear GPU cache and run garbage collector to free up memory.
            torch.cuda.empty_cache()
            gc.collect()

        print("Reference Example: ", ref_sentence)
        print("Prediction Example: ", sentence)

    # --- Calculate Corpus-Level Metrics ---
    bleu = sacrebleu.corpus_bleu(all_hypotheses, [all_references]).score
    chrf = sacrebleu.corpus_chrf(all_hypotheses, [all_references]).score
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1, rouge2, rougeL = [], [], []

    for ref, hyp in zip(all_references, all_hypotheses):
        scores = scorer.score(ref, hyp)
        rouge1.append(scores['rouge1'].fmeasure)
        rouge2.append(scores['rouge2'].fmeasure)
        rougeL.append(scores['rougeL'].fmeasure)
    avg_rouge1 = sum(rouge1) / len(rouge1) if rouge1 else 0.0
    avg_rouge2 = sum(rouge2) / len(rouge2) if rouge2 else 0.0
    avg_rougeL = sum(rougeL) / len(rougeL) if rougeL else 0.0

    # Final token-level accuracy.
    token_accuracy = total_correct_tokens / total_tokens if total_tokens > 0 else 0.0

    # Final perplexity.
    ppl = math.exp(total_loss / total_ppl_tokens)

    # --- Token-level Precision/Recall/F1/F0.5 ---
    y_true_tokens = []
    y_pred_tokens = []
    for ref, hyp in zip(all_references, all_hypotheses):
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()
        min_len = min(len(ref_tokens), len(hyp_tokens))
        # confronto solo fino alla lunghezza minima (per allineare bene)
        y_true_tokens.extend(ref_tokens[:min_len])
        y_pred_tokens.extend(hyp_tokens[:min_len])

    if len(y_true_tokens) > 0:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_tokens, y_pred_tokens, average="micro", zero_division=0
        )
        f05 = (1.25 * precision * recall) / (0.25 * precision + recall + 1e-8)
    else:
        precision = recall = f1 = f05 = 0.0


    # Character Error Rate (CER), Word Error Rate (WER), and Sentence Error Rate (SER).
    cer = jiwer.cer(all_references, all_hypotheses)
    wer = jiwer.wer(all_references, all_hypotheses)
    ser = sum(int(h != r) for h, r in zip(all_hypotheses, all_references)) / len(
        all_references) if all_references else 0.0

    # Average GLEU score across the corpus.
    gleu = sum(gleu_scores) / len(gleu_scores) if gleu_scores else 0.0
    

    if builder.bpe == 1:
        # Final token-level accuracy (as percentage)
        token_accuracy = (total_correct_tokens / total_tokens if total_tokens > 0 else 0.0)

        # Final perplexity.
        ppl = math.exp(total_loss / total_ppl_tokens) if total_ppl_tokens > 0 else float('inf')

        # --- Word-level Precision/Recall/F1/F0.5 via LCS ---
        precision = pr_matches / pr_hyp_len if pr_hyp_len > 0 else 0.0
        recall = pr_matches / pr_ref_len if pr_ref_len > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall + 1e-8) if (precision + recall) > 0 else 0.0
        f05 = (1.25 * precision * recall) / (0.25 * precision + recall + 1e-8) if (precision + recall) > 0 else 0.0

        # Character Error Rate (CER), Word Error Rate (WER), and Sentence Error Rate (SER).
        cer = jiwer.cer(all_references, all_hypotheses)
        wer = jiwer.wer(all_references, all_hypotheses)
        ser = sum(int(h != r) for h, r in zip(all_hypotheses, all_references)) / len(
        all_references) if all_references else 0.0

        # Average GLEU score across the corpus (as percentage)
        gleu = (sum(gleu_scores) / len(gleu_scores) if gleu_scores else 0.0)
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
        'cer': cer,
        'wer': wer,
        'ser': ser,
        'gleu': gleu
    }

def beamSearch(model, source, progress_bar, beam_width, builder, max_output_length=100):
    """
        Performs beam search decoding to generate sequences.

        Args:
            model (torch.nn.Module): The trained sequence-to-sequence model.
            source (torch.Tensor): The input source tensor of shape [batch_size, src_len].
            progress_bar: Unused parameter.
            beam_width (int): The number of beams to maintain during the search.
            builder (BuildDictionary_Map): The data builder instance for special token IDs.
            max_output_length (int): The maximum length of the generated sequence.

        Returns:
            list: A list of decoded sequences (each sequence is a list of token IDs).
    """
    batch_size = source.size(0)
    device = source.device
    sequences = torch.full((batch_size, beam_width, 1), builder.targetSOS, dtype=torch.long, device=device)
    scores = torch.zeros((batch_size, beam_width), dtype=torch.float, device=device)

    # Main decoding loop, runs for a maximum of `max_output_length` steps.
    for _ in range(max_output_length):
        num_candidates = sequences.size(1)
        input_seq = sequences.view(batch_size * num_candidates, -1)
        source_rep = source.unsqueeze(1).repeat(1, num_candidates, 1).view(batch_size * num_candidates, -1)
        # Get model logits for the next token prediction.
        _, logits = model(source_rep, input_seq)
        log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)

        # --- Expand and Find Top Candidates ---
        # Add the new log probabilities to the cumulative scores of the existing sequences.
        expanded_scores = scores.view(-1,1) + log_probs
        expanded_scores = expanded_scores.view(batch_size, -1)
        top_scores, top_indices = torch.topk(expanded_scores, beam_width, dim=1)

        # Determine which beam and which token each top candidate came from.
        beam_idx = top_indices // log_probs.size(-1)
        token_idx = top_indices % log_probs.size(-1)

        # --- Reconstruct the beams for the next step ---
        new_seqs = []
        for i in range(batch_size):
            new_seqs.append([
                sequences[i, beam_idx[i,b]].tolist() + [token_idx[i,b].item()]
                for b in range(beam_width)
            ])
        sequences = torch.tensor(new_seqs, device=device)
        scores = top_scores

    # --- Final Selection ---
    # After the loop, select the best hypothesis for each batch item.
    # Normalize scores by sequence length to avoid favoring shorter sequences.
    lengths = sequences.size(-1)
    norm_scores = scores / lengths
    best_idx = norm_scores.argmax(dim=1)
    final = []
    for i in range(batch_size):
        seq = sequences[i, best_idx[i]].tolist()
        if builder.targetEOS in seq:
            seq = seq[:seq.index(builder.targetEOS)]
        final.append(seq)
    
    return final, None, None
