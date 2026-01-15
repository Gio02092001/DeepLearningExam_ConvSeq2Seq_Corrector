import os
import torch
import yaml
from pprint import pprint

def trova_migliori_configurazioni(models_dir):
    """
    Analizza le sottocartelle di una directory, legge il punteggio CHRF da best_model.pt,
    e restituisce le 5 migliori configurazioni da config.yaml.
    """
    all_results = []

    print(f"🔍 Analisi della directory: {models_dir}\n")

    # Itera su ogni elemento nella directory dei modelli
    for dir_name in os.listdir(models_dir):
        subdir_path = os.path.join(models_dir, dir_name)

        # Assicurati che sia una directory
        if not os.path.isdir(subdir_path):
            continue

        model_path = os.path.join(subdir_path, 'best_model.pt')
        config_path = os.path.join(subdir_path, 'config.yaml')

        # Controlla che i file necessari esistano
        if os.path.exists(model_path) and os.path.exists(config_path):
            try:
                # Carica il checkpoint del modello.
                # map_location='cpu' è una buona pratica se il modello è stato salvato su GPU.
                checkpoint = torch.load(model_path, map_location='cpu')

                # --- IMPORTANTE: ESTRAZIONE DELLO SCORE ---
                # Cerca di estrarre il punteggio CHRF.
                # Dovrai forse adattare la chiave ('chrf') a quella che hai usato tu.
                # Esempi comuni: 'chrf_score', 'best_chrf', checkpoint['metrics']['chrf'], ecc.
                chrf_score = checkpoint['best_metric_ChrF']

                if chrf_score is not None:
                    # Leggi il file di configurazione YAML
                    with open(config_path, 'r') as f:
                        config_data = yaml.safe_load(f)

                    # Salva le informazioni raccolte
                    all_results.append({
                        'score': chrf_score,
                        'config': config_data,
                        'path': config_path
                    })
                else:
                    print(f"⚠️  Attenzione: Chiave 'chrf' non trovata in {model_path}")

            except Exception as e:
                print(f"❌ Errore durante l'elaborazione della cartella {dir_name}: {e}")

    # Ordina i risultati in base allo score, dal più alto al più basso
    sorted_results = sorted(all_results, key=lambda x: x['score'], reverse=True)

    # Prendi solo le prime 5 configurazioni
    top_5_results = sorted_results[:]

    # Stampa i risultati in un formato chiaro
    print("\n🏆 Le Migliori Configurazioni Trovate 🏆")
    print("=" * 40)
    for i, result in enumerate(sorted_results):
        print(f"\n--- 🏅 POSIZIONE #{i+1} 🏅 ---")
        print(f"🎯 Punteggio CHRF: {result['score']:.4f}")
        print(f"📄 File di Configurazione: {result['path']}")
        print("\n--- Contenuto di config.yaml: ---")
        # Usa pprint per una stampa più leggibile del dizionario
        print(result['config'])
        print("-" * 40)


if __name__ == '__main__':
    # Percorso della directory principale che contiene tutte le cartelle dei modelli
    main_models_directory = '/data01/dl24giocar/DeepLearningExam_ConvSeq2Seq_Corrector/models/BPETokenization/20KBPETokenization/BigRange'
    trova_migliori_configurazioni(main_models_directory)
