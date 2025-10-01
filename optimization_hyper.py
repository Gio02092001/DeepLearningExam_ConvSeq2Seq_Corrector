import json
import sys
import optuna
import subprocess
import re
import yaml
from optuna.pruners import MedianPruner

CONFIG_PATH = "Config/config.yaml"

def update_config(trial):
    """
    Updates the 'config.yaml' file with hyperparameter values suggested by an Optuna trial.

    This function is called at the beginning of each trial to set up the parameters
    for the corresponding training run.

    Args:
        trial (optuna.trial.Trial): The Optuna trial object which suggests hyperparameter values.
    """
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    # Suggest new hyperparameter values for the current trial.
    # Optuna will pick values from the specified ranges and distributions.
    config["learning_rate"] = trial.suggest_loguniform("learning_rate", 0.22, 0.3)
    config["beamWidth"] = trial.suggest_int("beamWidth", 6, 10)
    config["p_dropout"] = trial.suggest_uniform("p_dropout", 0.2, 0.35)
    x = trial.suggest_int("hidden_dim", 500, 950)
    config["hidden_dim"] = x
    config["embedding_dim"] = x
    config["encoderLayer"] = trial.suggest_int("encoderLayer", 2, 8)
    config["decoderLayer"] = trial.suggest_int("decoderLayer", 2, 8)
    config["batchSize"] = trial.suggest_int("batchSize", 32, 128)
    config['dataSet_probability'] = trial.suggest_uniform("dataSet_probability", 0.14, 0.18)
    config['dataSet_repetition'] = trial.suggest_int("dataSet_repetition", 3, 5)
    
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(config, f)

def run_trial(trial):
    """
    Executes a single training trial and reports the performance metric (CHR-F) to Optuna.

    This function serves as the 'objective function' for the Optuna study. It:
    1. Updates the config file with suggested parameters.
    2. Runs the main training script ('main.py') as a subprocess.
    3. Parses the output of the script in real-time to extract the CHR-F score after each epoch.
    4. Reports the score to Optuna, which may decide to 'prune' (stop) the trial early if it's unpromising.
    5. Implements a hard stop after 5 epochs to save time.

    Args:
        trial (optuna.trial.Trial): The Optuna trial object for the current run.

    Returns:
        float: The best CHR-F score achieved during the trial. Optuna will try to maximize this value.
    """
    update_config(trial)

    cmd = [sys.executable, "main.py"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    best_chrf = 0.0
    current_epoch = 0  # Inizializziamo l'epoca a 0

    for line in proc.stdout:
        print(line, end="")

        # 1. CERCA LA RIGA DI VALIDAZIONE PER LE AZIONI PRINCIPALI
        if "Validation — BLEU:" in line:
            match_chrf = re.search(r"CHR-F:\s*([0-9.]+)", line)
            if match_chrf:
                chrf = float(match_chrf.group(1))
                if chrf > best_chrf:
                    best_chrf = chrf

                # Fai il report a Optuna per l'epoca appena conclusa
                trial.report(chrf, current_epoch)

                # Controlla se Optuna vuole fare il pruning
                if trial.should_prune():
                    print(f"⏹️ Pruning at epoch {current_epoch}, CHR-F={chrf}")
                    proc.terminate()
                    proc.wait()
                    raise optuna.TrialPruned()

            # 2. DOPO AVER PROCESSATO LA VALIDAZIONE, CONTROLLA LO STOP ANTICIPATO
            if current_epoch >= 5:
                print(f"🔴 Early stopping: Trial finished after validation for epoch {current_epoch}")
                proc.terminate()
                break  # Esci dal ciclo di lettura dell'output

        # 3. AGGIORNA SOLO LO STATO QUANDO TROVI LA FINE DI UN'EPOCA
        match_epoch = re.search(r"Epoch\s+(\d+)\s+finished", line)
        if match_epoch:
            current_epoch = int(match_epoch.group(1))

    proc.wait()
    return best_chrf
    """
    update_config(trial)

    # Define the command to execute the main training script using the same Python interpreter.
    cmd = [sys.executable, "main.py"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    best_chrf = 0.0
    # Read the output from the subprocess line by line, in real-time.
    for line in proc.stdout:
        print(line, end="")
        # Check if an epoch has finished using a regular expression.
        match = re.search(r"Epoch\s+(\d+)\s+finished", line)
        if match:
            current_epoch = int(match.group(1))
            # Early stopping condition: terminate the trial after 5 epochs to speed up the search.
            if current_epoch >= 5:
                print(f"🔴 Epoch {current_epoch} finished → stopping trial early")
                proc.terminate()
                break

        if "Validation — BLEU:" in line:
            # Use a regular expression to find and extract the CHR-F score.
            match = re.search(r"CHR-F:\s*([0-9.]+)", line)
            if match:
                chrf = float(match.group(1))
                # Keep track of the best CHR-F score seen so far in this trial.
                if chrf > best_chrf:
                    best_chrf = chrf

                trial.report(chrf, current_epoch)
                # Check if the pruner recommends stopping this trial.
                if trial.should_prune():
                    print(f"⏹️ Pruning at epoch {current_epoch}, CHR-F={chrf}")
                    proc.terminate()
                    proc.wait()
                    raise optuna.TrialPruned()

    proc.wait()
    return best_chrf  # Optuna massimizza questa metrica
    """
if __name__ == "__main__":
    # --- Optuna Study Setup and Execution ---
    study = optuna.create_study(
        direction="maximize",
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    )
    study.optimize(run_trial, n_trials=50)

    # --- Results Reporting ---
    print("Miglior trial:")
    print(study.best_trial.params)
    print("CHR-F migliore:", study.best_value)
    # Mostra le top 3 trial

    # Sort all trials by their final value in descending order and get the top 3.
    top_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:3]
    print("\nTop 3 trial:")
    for i, trial in enumerate(top_trials, 1):
        print(f"Rank {i}: CHR-F={trial.value}")
        print(trial.params)

    # --- Save Results to a JSON file ---
    results = {
        "best_trial": {
            "params": study.best_trial.params,
            "CHR-F": study.best_value
        },
        "top_3_trials": [
            {"rank": i + 1, "CHR-F": t.value, "params": t.params}
            for i, t in enumerate(top_trials)
        ]
    }
    with open("Results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Results saved to Results.json")
