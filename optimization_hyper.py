import sys

import optuna
import subprocess
import re
import yaml
from optuna.pruners import MedianPruner

CONFIG_PATH = "Config/config.yaml"

def update_config(trial):
    """Aggiorna il config.yaml con i valori suggeriti da Optuna."""
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    config["learning_rate"] = trial.suggest_loguniform("learning_rate", 0.2, 0.3)
    config["beamWidth"] = trial.suggest_int("beamWidth", 3, 7)
    config["p_dropout"] = trial.suggest_uniform("p_dropout", 0.05, 0.5)
    x = trial.suggest_int("hidden_dim", 256, 1028)
    config["hidden_dim"] = x
    config["embedding_dim"] = x
    config["encoderLayer"] = trial.suggest_int("encoderLayer", 5, 15)
    config["decoderLayer"] = trial.suggest_int("decoderLayer", 5, 15)
    config["batchSize"] = trial.suggest_int("batchSize", 32, 128)
    config['dataSet_probability']= trial.suggest_uniform("dataSet_probability", 0.05, 0.18)
    config['dataSet_repetition'] = trial.suggest_int("dataSet_probability", 2,6 )


    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(config, f)

def run_trial(trial):
    """Lancia main.py e legge la metrica CHR-F dall’output."""
    update_config(trial)

    cmd = [sys.executable, "main.py"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    best_chrf = 0.0
    for line in proc.stdout:
        print(line, end="")
        # 🔹 Detect finished epoch
        match = re.search(r"Epoch\s+(\d+)\s+finished", line)
        if match:
            current_epoch = int(match.group(1))
            if current_epoch >= 5:
                print(f"🔴 Epoch {current_epoch} finished → stopping trial early")
                proc.terminate()
                break

        if "Validation — BLEU:" in line:
            match = re.search(r"CHR-F:\s*([0-9.]+)", line)
            if match:
                chrf = float(match.group(1))
                if chrf > best_chrf:
                    best_chrf = chrf

                trial.report(chrf, current_epoch)
                if trial.should_prune():
                    print(f"⏹️ Pruning at epoch {current_epoch}, CHR-F={chrf}")
                    proc.terminate()
                    proc.wait()
                    raise optuna.TrialPruned()

    proc.wait()
    return best_chrf  # Optuna massimizza questa metrica

if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    )
    study.optimize(run_trial, n_trials=40)

    print("Miglior trial:")
    print(study.best_trial.params)
    print("CHR-F migliore:", study.best_value)
    # Mostra le top 3 trial

    top_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:3]
    print("\nTop 3 trial:")
    for i, trial in enumerate(top_trials, 1):
        print(f"Rank {i}: CHR-F={trial.value}")
        print(trial.params)
