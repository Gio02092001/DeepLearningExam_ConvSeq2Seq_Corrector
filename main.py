import os

import torch
import yaml
from sacremoses import MosesTokenizer
from torch.optim.lr_scheduler import StepLR, LambdaLR
from tqdm import tqdm

from Model_New.ModelBuilder_new import ConvModel_New
from BuildDictionary_Map import BuildDictionary_Map
from train import train


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


def main():
    """
    Main function to build and execute all model functions.
    """
    tqdm.write("--------------START SETUP----------------------")
    device = torch.device("cuda:1" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))


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

    ckpt = torch.load("1751611289_best_model_epoch73.pt", map_location="cuda:1" )
    model.load_state_dict(ckpt["model_state"])

    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=config["learning_rate"], momentum=config["nestorovsMomentum"], nesterov=True
    )
    optimizer.load_state_dict(ckpt["optimizer_state"])

    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 0.1 ** step)

    # Execute model training
    train(model, optimizer, scheduler, train_data, builder, word_dict, config["renormalizationLimit"],
          config["maximumlearningRateLimit"], target_word_dict, validation_data, config["fixedNumberOfInputElements"],config["batchSize"], index_to_target_word_dict, config['patience'], index_to_word_dict)


if __name__ == "__main__":
    main()
