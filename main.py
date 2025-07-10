import os

import torch
import yaml
from sacremoses import MosesTokenizer
from torch.optim.lr_scheduler import StepLR, LambdaLR
from tqdm import tqdm

from Model_New.ModelBuilder_new import ConvModel_New
from BuildDictionary_Map import BuildDictionary_Map
from train import train
import pynvml
import argparse


def load_parameters(config_path="Config/config.yaml"):
    """
    Load model parameters from a YAML config file.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_dataset(builder, config):
    """
    Load precomputed dictionaries and split the dataset.
    """
    word_dict, target_word_dict, sentence_map, index_to_target_word_dict, index_to_word_dict = builder.loadDictionaries(
        config["dataSet_Sentence"], config["dataSet_repetition"], config["dataSet_probability"]
    )

    vocab_size = len(word_dict)+1
    target_vocab_size = len(target_word_dict)+1

    tqdm.write(f"Vocabulary size: {vocab_size}")
    tqdm.write(f"Target vocabulary size: {target_vocab_size}")

    train_data, validation_data = builder.splitSet(sentence_map, config["validationSet"])

    return word_dict, target_word_dict, sentence_map, vocab_size, target_vocab_size, train_data, validation_data, index_to_target_word_dict, index_to_word_dict

def pick_best_gpu():
    if not torch.cuda.is_available():
        return None


    # proviamo a usare NVIDIA NVML per leggere memoria libera

    pynvml.nvmlInit()
    n_gpus = torch.cuda.device_count()

    best_idx = None
    best_free = 0

    for i in range(n_gpus):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        free_mb = info.free
        print(f"GPU {i}: {free_mb:.1f} liberi")
        if free_mb > best_free:
            best_free = free_mb
            best_idx = i

    pynvml.nvmlShutdown()

    return best_idx



def main():
    """
    Main function to build and execute all model functions.
    """
    tqdm.write("--------------START SETUP----------------------")
    pretrained=None
    # scelta del device
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
    #device = torch.device("cuda:1" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', help="Percorso del file")
    args = parser.parse_args()

    if hasattr(args, 'pretrained') and args.pretrained is not None:
        # args.file esiste e non è None
        pretrained = args.pretrained
        ckpt = torch.load(f"/models/{pretrained}/best_model.pt", map_location=device)
        config = load_parameters("/models/pretrained/config.yaml")
        # fai qualcosa con file_path
    else:
        config = load_parameters()


    # Scommenta questi per creare gli ultimi dizionari grossi
    builder = BuildDictionary_Map(config["dataSet_Sentence"], config["dataSet_repetition"],
                                  config["dataSet_probability"])

    if not os.path.exists(f'data/dictionaries/{config["dataSet_Sentence"]}x{config["dataSet_repetition"]}x{config["dataSet_probability"]}_SentenceMap.pkl'):
        builder.buildDictionary()



    word_dict, target_word_dict, sentence_map, vocab_size, target_vocab_size, train_data, validation_data, index_to_target_word_dict, index_to_word_dict = load_dataset(
        builder, config)


    # Initialize Model_Old, Optimizer, and Scheduler
    model = ConvModel_New(
        target_vocab_size, vocab_size, config["fixedNumberOfInputElements"], config["embedding_dim"],
        config["p_dropout"], config["hidden_dim"], config["kernel_width"], config["encoderLayer"],
        config["decoderLayer"], builder.sourceUNK, device
    )


    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=config["learning_rate"], momentum=config["nestorovsMomentum"], nesterov=True
    )

    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 0.1 ** step)

    if pretrained is not None:
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])

    # Execute model training
    train(model, optimizer, scheduler, train_data, builder, word_dict, config["renormalizationLimit"],
          config["maximumlearningRateLimit"], target_word_dict, validation_data, config["fixedNumberOfInputElements"],config["batchSize"], index_to_target_word_dict, config['patience'], index_to_word_dict, ckpt, pretrained)


if __name__ == "__main__":
    main()
