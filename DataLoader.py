from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    """Custom dataset for translation pairs"""

    def __init__(self, data_dict, word_dict, target_word_dict, builder, tokenize_fn):
        self.data = []
        self.tokenize_fn = tokenize_fn
        self.word_dict = word_dict
        self.target_word_dict=target_word_dict
        self.builder = builder

        # Preprocess all data
        for source, target in data_dict.items():
            # Tokenize
            source_tokens = self.tokenize_fn.tokenize(source)
            target_tokens = self.tokenize_fn.tokenize(target)

            # Convert to indices
            source_indices = [word_dict.get(token) for token in source_tokens] + [builder.sourceEOS]
            target_indices = [builder.targetSOS] + [target_word_dict.get(token) for token in target_tokens]
            self.data.append((source_indices, target_indices))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]