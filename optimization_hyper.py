import optuna
import subprocess
import re
import yaml

CONFIG_PATH = "Config/config.yaml"

def update_config(trial):
    """Aggiorna il config.yaml con i valori suggeriti da Optuna."""
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    config["learning_rate"] = trial.suggest_loguniform("learning_rate", 0.2, 0.3)
    config["beamWidth"] = trial.suggest_int("beamWidth", 3, 7)
    config["p_dropout"] = trial.suggest_uniform("p_dropout", 0.05, 0.5)
    x = trial.suggest_int("hidden_dim", 128, 1024)
    config["hidden_dim"] = x
    config["embedding_dim"] = x
    config["encoderLayer"] = trial.suggest_int("encoderLayer", 5, 15)
    config["decoderLayer"] = trial.suggest_int("decoderLayer", 5, 15)
    config["batch_size"] = trial.suggest_int("batch_size", 32, 128)

    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(config, f)

def run_trial(trial):
    """Lancia main.py e legge la metrica CHR-F dall’output."""
    update_config(trial)

    cmd = ["python", "main.py"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    best_chrf = 0.0
    for line in proc.stdout:
        print(line, end="")
        if "Validation — BLEU:" in line:
            match = re.search(r"CHR-F:\s*([0-9.]+)", line)
            if match:
                chrf = float(match.group(1))
                if chrf > best_chrf:
                    best_chrf = chrf

    proc.wait()
    return best_chrf  # Optuna massimizza questa metrica

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(run_trial, n_trials=20)

    print("Miglior trial:")
    print(study.best_trial.params)
    print("CHR-F migliore:", study.best_value)
    # Mostra le top 3 trial

    top_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:3]
    print("\nTop 3 trial:")
    for i, trial in enumerate(top_trials, 1):
        print(f"Rank {i}: CHR-F={trial.value}")
        print(trial.params)
