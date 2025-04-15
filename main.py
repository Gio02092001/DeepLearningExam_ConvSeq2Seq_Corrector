import torch
import yaml
from sacremoses import MosesTokenizer
from torch.optim.lr_scheduler import StepLR

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
    word_dict, target_word_dict, sentence_map = builder.loadDictionaries(
        config["dataSet_Sentence"], config["dataSet_repetition"], config["dataSet_probability"]
    )

    vocab_size = len(word_dict)
    target_vocab_size = len(target_word_dict)

    print(f"Vocabulary size: {vocab_size}")
    print(f"Target vocabulary size: {target_vocab_size}")

    train_data, validation_data = builder.splitSet(sentence_map, config["validationSet"])

    return word_dict, target_word_dict, sentence_map, vocab_size, target_vocab_size, train_data, validation_data


def main():
    """
    Main function to build and execute all model functions.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_parameters()
    builder = BuildDictionary_Map()


    word_dict, target_word_dict, sentence_map, vocab_size, target_vocab_size, train_data, validation_data = load_dataset(
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
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

    # Execute model training
    train(model, optimizer, scheduler, train_data, builder, word_dict, config["renormalizationLimit"],
          config["maximumlearningRateLimit"], target_word_dict, validation_data)


if __name__ == "__main__":
    #Scommenta questi per creare gli ultimi dizionari grossi
    #builder=BuildDictionary_Map()
    #builder.buildDictionary()

    main()
