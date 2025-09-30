import os
import time
import shutil
import torch
import yaml
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from Model_New.ModelBuilder_new import ConvModel_New
from BuildDictionary_Map import BuildDictionary_Map
from train import train
import pynvml
import argparse

def load_parameters(config_path="Config/config.yaml"):
    """
    Loads model and training parameters from a YAML configuration file.

    Args:
        config_path (str): The path to the YAML configuration file.

    Returns:
        dict: A dictionary containing the configuration parameters.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_dataset(builder, config, timestamp):
    """
    Loads the dataset components: dictionaries, sentence map, and splits it into train/validation sets.

    Args:
        builder (BuildDictionary_Map): An instance of the data builder class.
        config (dict): The configuration dictionary.
        timestamp: An identifier for the current run.

    Returns:
        tuple: A tuple containing all necessary data components for the model, including
               vocabularies, the full sentence map, vocab sizes, train/validation data splits,
               and index-to-word mappings.
    """
    word_dict, target_word_dict, sentence_map, index_to_target_word_dict, index_to_word_dict = builder.loadDictionaries(
        config["dataSet_Sentence"], config["dataSet_repetition"], config["dataSet_probability"], timestamp
    )

    vocab_size = len(word_dict)+1
    target_vocab_size = len(target_word_dict)+1
    tqdm.write(f"Vocabulary size: {vocab_size}")
    tqdm.write(f"Target vocabulary size: {target_vocab_size}")
    train_data, validation_data = builder.splitSet(sentence_map, config["validationSet"])

    return word_dict, target_word_dict, sentence_map, vocab_size, target_vocab_size, train_data, validation_data, index_to_target_word_dict, index_to_word_dict

def pick_best_gpu():
    """
        Selects the CUDA-enabled GPU with the most available free memory.

        This function uses the pynvml library to query the memory status of all available
        NVIDIA GPUs and returns the index of the one with the highest free memory.

        Returns:
            int or None: The index of the best GPU, or None if no CUDA devices are available.
    """
    if not torch.cuda.is_available():
        return None

    # Initialize the NVIDIA Management Library (NVML).
    pynvml.nvmlInit()
    n_gpus = torch.cuda.device_count()

    best_idx = None
    best_free = 0

    # Iterate through all available GPUs.
    for i in range(n_gpus):
        # Get information for the GPU.
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        free_mb = info.free
        print(f"GPU {i}: {free_mb:.1f} liberi")

        # If the current GPU has more free memory than the best one found so far, update it.
        if free_mb > best_free:
            best_free = free_mb
            best_idx = i

    pynvml.nvmlShutdown()
    return best_idx

def main():
    """
    The main execution function for the entire model pipeline.

    This function handles:
    1. Setting up the computation device (GPU/CPU/MPS).
    2. Loading configuration from a file or a pre-trained model checkpoint.
    3. Building or loading the dataset.
    4. Initializing the model, optimizer, and learning rate scheduler.
    5. Starting the training process.
    """

    tqdm.write("--------------START SETUP----------------------")
    pretrained=None
    ckpt=None

    # --- Device Selection ---
    # Pick the best available GPU based on free memory.
    best_gpu = pick_best_gpu()
    if best_gpu is not None:
        device = torch.device(f"cuda:{best_gpu}")
        print(f"Usando CUDA device gpu:{best_gpu}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Usando MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Usando CPU")

    # --- Argument Parsing and Configuration Loading ---
    # Set up an argument parser to handle command-line options.
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', help="Percorso del file")
    args = parser.parse_args()

    # Check if a pretrained model path was provided.
    if hasattr(args, 'pretrained') and args.pretrained is not None:
        pretrained = args.pretrained
        ckpt = torch.load(f"models/{pretrained}/last_model.pt", map_location=device)
        config = load_parameters(f"models/{pretrained}/config.yaml")
        timestamp = pretrained
        learning_rate = ckpt['learning_rate']
    else:
        # This block runs for a NEW training session.
        config = load_parameters()
        timestamp = str(int(time.time()))
        os.mkdir(f"models/{timestamp}")
        src_path = "Config/config.yaml"
        dst_path = f"models/{timestamp}"
        shutil.copy2(src_path, dst_path)
        learning_rate=config["learning_rate"]

    # --- Dataset Generation and Loading ---
    # Initialize the data builder with parameters from the config.
    builder = BuildDictionary_Map(config["dataSet_Sentence"], config["dataSet_repetition"],
                                  config["dataSet_probability"], config["BPE"])

    # Check if the dataset needs to be built.
    if config["BPE"]==0:
        if not os.path.exists(f'data/dictionaries/{config["dataSet_Sentence"]}x{config["dataSet_repetition"]}x{config["dataSet_probability"]}_SentenceMap.pkl'):
            builder.buildDictionary()
    else:
        if not os.path.exists(f'data/dictionaries/{config["dataSet_Sentence"]}x{config["dataSet_repetition"]}x{config["dataSet_probability"]}_SentenceMap_BPE.pkl'):
            builder.buildDictionary()

    # Load all dataset components (dictionaries, splits, etc.).
    word_dict, target_word_dict, sentence_map, vocab_size, target_vocab_size, train_data, validation_data, index_to_target_word_dict, index_to_word_dict = load_dataset(
        builder, config, timestamp)

    # --- Model, Optimizer, and Scheduler Initialization ---
    # Initialize the convolutional sequence-to-sequence model with parameters from the config.
    model = ConvModel_New(
        target_vocab_size, vocab_size, config["fixedNumberOfInputElements"], config["embedding_dim"],
        config["p_dropout"], config["hidden_dim"], config["kernel_width"], config["encoderLayer"],
        config["decoderLayer"], builder.sourceUNK, device
    )
    model.to(device)

    # Initialize the SGD optimizer with Nesterov momentum.
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=config["nestorovsMomentum"], nesterov=True
    )

    # Initialize a learning rate scheduler that reduces the LR by a factor of 0.1 at each step.
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 0.1 ** step)

    # If resuming training, load the saved state for the model and optimizer.
    if pretrained is not None:
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])

    # --- Training Execution ---
    # Call the main training function to start the training loop.
    train(model, optimizer, scheduler, train_data, builder, word_dict, config["renormalizationLimit"],
          config["maximumlearningRateLimit"], target_word_dict, validation_data, config["fixedNumberOfInputElements"],config["batchSize"], index_to_target_word_dict, config['patience'], index_to_word_dict,timestamp, ckpt, pretrained)

if __name__ == "__main__":
    main()
