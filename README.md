# DeepLearningExam_ConvSeq2Seq_Corrector

## About The Project

This project is an implementation of a Convolutional Sequence-to-Sequence (ConvS2S) model, based on the architecture described in **"Convolutional Sequence to Sequence Learning"** by Gehring et al. (2017). The model is designed to correct artificially corrupted sentences, simulating common typing errors.

This work was developed as part of the Deep Learning course exam at the **University of Florence**, under the supervision of **Professor Paolo Frasconi**.

The core task is to take a "noisy" or corrupted sentence as input and have the model output the original, correct version.

## Technology Stack

- **PyTorch** - Deep learning framework
- **Optuna** - Hyperparameter optimization with MedianPruner for early trial stopping
- **NLTK & Hugging Face Tokenizers** - Text preprocessing and tokenization
- **SacreBLEU & Jiwer** - Evaluation metrics (BLEU score and Word Error Rate)
- **PyYAML** - Configuration file management
- **TensorBoard** - Training visualization and metric logging

## Dataset

The model is trained on artificially corrupted English sentences derived from the **enwik9** corpus (first 1GB of English Wikipedia XML dump).

### Data Directory Structure

```
data/
├── enwik9              # Wikipedia text corpus
└── Secondwikifil.pl    # Perl preprocessing script
```

The `enwik9` dataset is a standard benchmark corpus commonly used for text compression and language modeling tasks. It can be downloaded from: http://mattmahoney.net/dc/textdata.html

### Data Processing Pipeline

The `BuildDictionary_Map.py` script processes the raw corpus to:

1. **Extract sentences** from the Wikipedia XML dump
2. **Tokenize** using either word-level or BPE (Byte-Pair Encoding) tokenization
3. **Generate corrupted versions** by simulating typing errors:
   - Character insertions
   - Character deletions
   - Character substitutions
   - Character swaps
4. **Build vocabularies** for both source (corrupted) and target (original) text
5. **Create training/validation splits** based on configured ratios
6. **Save processed data** as `.pkl` files for efficient loading

The corruption process parameters (probability, repetitions, number of sentences) are configurable in `Config/config.yaml`:
- `dataSet_Sentence`: Number of sentences to process
- `dataSet_probability`: Probability of corruption per character
- `dataSet_repetition`: Number of corrupted versions per original sentence

### On-the-Fly Data Augmentation

During training, an additional **2% corruption probability** is applied on-the-fly to each batch to improve model robustness and generalization.

## Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

- Python 3.10.12
- pip package manager
- CUDA-compatible GPU (recommended for training)

### Installation

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/Gio02092001/DeepLearningExam_ConvSeq2Seq_Corrector.git
   cd DeepLearningExam_ConvSeq2Seq_Corrector
   ```

2. Install all the required Python packages using the requirements.txt file:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the dataset is available in the `data/` directory. If not present, download `enwik9` from the source mentioned above.

## Project Structure

```
DeepLearningExam_ConvSeq2Seq_Corrector/
├── Config/
│   └── config.yaml           # Configuration file with hyperparameters
├── Model_New/                # Model architecture files
│   ├── Attention_New.py      # Attention mechanism implementation
│   ├── Classification_New.py # Classification/output layer
│   ├── Decoder_New.py        # Decoder module
│   ├── Encoder_New.py        # Encoder module
│   ├── InitialEmbedding_new.py  # Embedding layer
│   ├── LayerModules_new.py   # Convolutional layer modules
│   └── ModelBuilder_new.py   # Main model builder
├── data/                     # Dataset directory
│   ├── enwik9                # Wikipedia corpus
│   └── Secondwikifil.pl      # Preprocessing script
├── models/                   # Saved model checkpoints (generated during training)
├── runs/                     # TensorBoard logs (generated during training)
├── BuildDictionary_Map.py    # Dictionary building and data corruption utilities
├── DataLoader.py             # Data loading and batch preprocessing
├── main.py                   # Main training script
├── train.py                  # Training loop implementation
├── validation.py             # Validation and evaluation metrics
├── optimization_hyper.py     # Optuna hyperparameter optimization
├── findBest.py               # Find best model from optimization trials
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Usage

There are two main ways to run this project: standard training or hyperparameter optimization.

### 1. Standard Training

To train the model with the current settings, run `main.py`. The model will use the parameters defined in the configuration file.

```bash
python3 main.py
```

**Configuration**: To change hyperparameters such as learning rate, batch size, embedding dimensions, or dropout, you can directly edit the `Config/config.yaml` file.

**Checkpoints**: The training script will save checkpoints to `models/{timestamp}/`:
- `best_model.pt` - Model with the best validation CHR-F score
- `last_model.pt` - Model from the most recent epoch

Each checkpoint contains:
- Model state
- Optimizer state
- Best metric (CHR-F score)
- Current epoch number
- Learning rate
- Early stopping counter
- Global training step

**TensorBoard Logging**: Monitor training progress in real-time:
```bash
tensorboard --logdir=runs/{timestamp}
```

Logged metrics include:
- Training loss and accuracy per batch
- Gradient norms
- Learning rate schedule
- All 15 validation metrics per epoch

### 2. Hyperparameter Optimization

To start a hyperparameter search using Optuna, run the `optimization_hyper.py` script. This will automatically search for the best hyperparameter combination.

```bash
python3 optimization_hyper.py
```

#### Optimization Details

**Objective Metric**: CHR-F score (maximized)

**Optimization Strategy**:
- **Number of Trials**: 100
- **Pruning**: MedianPruner with 3 startup trials and 1 warmup step
- **Early Stopping**: Each trial is automatically stopped after 3 epochs to save time
- **Real-time Monitoring**: CHR-F score is parsed and reported to Optuna after each epoch
- **Automatic Pruning**: Unpromising trials are terminated early based on intermediate results

**Results**: At the end of the optimization run:
- Best trial parameters and CHR-F score are displayed
- Top 3 trials are shown with their parameters and scores
- All results are saved to `Results.json` containing:
  - Best trial details
  - Top 3 trials ranking

Example `Results.json` structure:
```json
{
  "best_trial": {
    "params": {
      "learning_rate": 0.315,
      "p_dropout": 0.357,
      "hidden_dim": 946,
      ...
    },
    "CHR-F": 85.42
  },
  "top_3_trials": [
    {"rank": 1, "CHR-F": 85.42, "params": { ... }},
    {"rank": 2, "CHR-F": 84.89, "params": { ... }},
    {"rank": 3, "CHR-F": 84.12, "params": { ... }}
  ]
}
```

## Training Process

### Data Processing
- Supports both **word-level** and **BPE (Byte-Pair Encoding)** tokenization
- Equal-length batch sampling for efficient GPU utilization
- Dynamic data corruption (2% probability) applied during training
- Automatic GPU selection based on available memory

### Optimization
- **Loss Function**: Cross-Entropy Loss (ignoring padding tokens)
- **Gradient Clipping**: Prevents gradient explosion using configurable max norm
- **Learning Rate Scheduling**: Reduces learning rate when validation performance plateaus
- **Optimizer**: Configurable (typically SGD with Nesterov momentum)

### Early Stopping
- **Primary Metric**: CHR-F score (character n-gram F-score)
- **Patience**: Configurable number of epochs without improvement before LR reduction
- Training continues until learning rate falls below `maximumlearningRateLimit`

## Model Architecture

The Convolutional Sequence-to-Sequence model consists of:

- **Encoder**: Multiple convolutional layers with residual connections
- **Decoder**: Convolutional layers with attention mechanism
- **Attention**: Multi-hop attention connecting encoder and decoder states
- **Position Embeddings**: Learned positional encodings for sequence information
- **Output Layer**: Linear projection to vocabulary with softmax

Key hyperparameters (configurable in `config.yaml`):
- `embedding_dim`: Dimension of word embeddings 
- `hidden_dim`: Hidden layer dimensions 
- `encoderLayer` / `decoderLayer`: Number of convolutional layers 
- `kernel_width`: Convolutional kernel size 
- `p_dropout`: Dropout probability 
- `learning_rate`: Initial learning rate 
- `batchSize`: Number of sequences per batch
- `patience`: Epochs to wait before reducing learning rate 
- `renormalizationLimit`: Maximum gradient norm for clipping 
- `beamWidth`: Beam width for beam search decoding 
- `fixedNumberOfInputElements`: Maximum sequence length 
- `BPE`: Use BPE tokenization (1) or word-level (0) 

## Evaluation Metrics

The model performance is evaluated using a comprehensive set of **15 metrics** implemented in `validation.py`:

### Translation Quality Metrics
- **BLEU Score** - Corpus-level BLEU score using SacreBLEU, measures n-gram precision
- **CHR-F Score** - Character n-gram F-score, more robust to morphological variations (used for early stopping and optimization)
- **GLEU Score** - Sentence-level GLEU using NLTK, balances precision and recall

### Error Rate Metrics
- **CER (Character Error Rate)** - Character-level error rate using Jiwer
- **WER (Word Error Rate)** - Word-level error rate using Jiwer
- **SER (Sentence Error Rate)** - Percentage of sentences that are not perfectly correct

### Token-Level Metrics
- **Token Accuracy** - Percentage of correctly predicted tokens (using LCS alignment for BPE mode)
- **Precision** - Token-level precision (micro-averaged)
- **Recall** - Token-level recall (micro-averaged)
- **F1 Score** - Harmonic mean of precision and recall
- **F0.5 Score** - Weighted F-score favoring precision over recall

### Semantic Similarity
- **ROUGE-1** - Unigram overlap F-measure
- **ROUGE-2** - Bigram overlap F-measure
- **ROUGE-L** - Longest common subsequence F-measure

### Language Modeling
- **Perplexity** - Exponential of cross-entropy loss, measures model confidence

All metrics are computed during validation using **beam search decoding** (beam width: 5) to ensure realistic model performance assessment without teacher forcing.

## Resuming Training

To resume training from a checkpoint:

```python
# In your code, load the checkpoint
ckpt = torch.load('models/{timestamp}/last_model.pt')
model.load_state_dict(ckpt['model_state'])
optimizer.load_state_dict(ckpt['optimizer_state'])
# Then pass ckpt to the train function
```

The training will automatically resume from the saved epoch with all states preserved (including early stopping counters and learning rate schedule).

## References

### Academic References

Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, and Yann N. Dauphin. 2017. 
**Convolutional Sequence to Sequence Learning.** 
In Proceedings of the 34th International Conference on Machine Learning (ICML 2017).
[Paper Link](https://arxiv.org/abs/1705.03122)

### Dataset Reference

Matt Mahoney. **Text Compression Benchmark (enwik9)**.  
Available at: http://mattmahoney.net/dc/textdata.html

### AI Tools Used

This project was developed with assistance from various AI-powered tools for debugging, code optimization, and documentation:

- **GitHub Copilot** - Code suggestions and autocompletion
- **ChatGPT** (OpenAI) - Debugging assistance, code explanation, and problem-solving
- **Claude** (Anthropic) - Code review and optimization suggestions
- **Gemini** (Google) - Additional research and documentation support

*Note: All AI-generated suggestions were reviewed, tested, and adapted by the author. The core implementation, architectural decisions, and experimental design remain the original work of the author under the supervision of Prof. Paolo Frasconi.*

### Further Details

Further Details can be found under Carlucci_Giovanni_ConvolutionalSeq2Seq_Presentazione.pdf

## Author

**Giovanni Carlucci** - University of Florence  
Deep Learning Course Exam - 2026

Contact: giovanni.carlucci@edu.unifi.it

## License

This project is developed for educational purposes as part of a university exam at the University of Florence.

## Acknowledgments

- Professor Paolo Frasconi for supervision and guidance
- University of Florence, Deep Learning Course
- The open-source community for the libraries and tools used in this project

---

**Note**: Remember to replace `[Your Last Name]` with your actual surname and optionally add your contact email before submission.
