# LSTM Word Embeddings for Sentiment Analysis

A deep learning project that implements Long Short-Term Memory (LSTM) networks with word embeddings for sentiment classification on movie reviews using the IMDB dataset.

## Overview

This project demonstrates how to build and train an LSTM neural network for binary sentiment classification. The model uses word embeddings to convert text data into dense vector representations and processes sequential data through LSTM layers to predict whether movie reviews are positive or negative.

## Features

- **Word Embeddings**: Custom embedding layer that learns dense representations of words
- **LSTM Architecture**: Sequential model with LSTM layers for processing text sequences
- **Binary Classification**: Predicts sentiment (positive/negative) of movie reviews
- **Visualization**: Training and validation loss/accuracy plots
- **Reproducible Results**: Fixed random seeds for consistent results

## Dataset

The project uses the **IMDB Movie Reviews Dataset** from Keras datasets:
- **Training samples**: 25,000 movie reviews
- **Test samples**: 25,000 movie reviews
- **Vocabulary size**: 20,000 most frequent words
- **Sequence length**: Padded to 25 words per review

## Model Architecture

```
Model: Sequential
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, None, 128)         2,560,000 
lstm (LSTM)                  (None, 128)               131,584   
dense (Dense)                (None, 1)                 129       
=================================================================
Total params: 2,691,713
Trainable params: 2,691,713
Non-trainable params: 0
```

### Layer Details:
1. **Embedding Layer**: Converts word indices to dense vectors (20,000 → 128 dimensions)
2. **LSTM Layer**: 128 units with 0.2 dropout for regularization
3. **Dense Layer**: Single output neuron with sigmoid activation for binary classification

## Requirements

```python
tensorflow>=2.0
keras
numpy
matplotlib
pandas
```

## Installation

1. Clone the repository:
```bash
git clone <https://github.com/0xafraidoftime/Sentiment-Analysis>
cd lstm-word-embeddings
```

2. Install dependencies:
```bash
pip install tensorflow numpy matplotlib pandas
```

## Usage

### Running the Script

Execute the Python script:
```bash
python lstm_word_embeddings.py
```

### Running the Jupyter Notebook

Open and run the Jupyter notebook:
```bash
jupyter notebook LSTM_Word_Embeddings.ipynb
```

## Training Configuration

- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy
- **Epochs**: 15
- **Batch Size**: 16
- **Validation Split**: 20%

## Results

The model generates:
- Training and validation loss curves
- Training and validation accuracy curves
- Final test accuracy and loss metrics

Expected performance:
- Training typically converges within 15 epochs
- Test accuracy generally ranges between 80-85%

## File Structure

```
├── lstm_word_embeddings.py     # Main Python script
├── LSTM_Word_Embeddings.ipynb  # Jupyter notebook version
├── README.md                   # This file
└── requirements.txt            # Dependencies (optional)
```

## Key Components

### Data Preprocessing
- Loads IMDB dataset with vocabulary limited to 20,000 words
- Pads sequences to uniform length of 25 words
- Splits data into training and test sets

### Model Training
- Uses binary crossentropy loss for sentiment classification
- Implements early regularization through dropout
- Validates on 20% of training data

### Visualization
- Plots training vs validation loss over epochs
- Plots training vs validation accuracy over epochs
- Helps identify overfitting and convergence

## Customization

You can modify several parameters to experiment with the model:

```python
# Vocabulary size
max_features = 20000  # Try 10000, 30000

# Sequence length
maxlen = 25  # Try 50, 100

# LSTM units
lstm_units = 128  # Try 64, 256

# Embedding dimensions
embedding_dim = 128  # Try 64, 256

# Training epochs
epochs = 15  # Try 10, 20

# Batch size
batch_size = 16  # Try 32, 64
```

## Reproducibility

The code sets random seeds for:
- Python `random` module
- NumPy random number generator
- TensorFlow random operations

This ensures consistent results across multiple runs.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- TensorFlow/Keras team for the framework and IMDB dataset
- Original IMDB dataset creators for providing the movie review data
- The deep learning community for LSTM architecture insights

## Future Improvements

- [ ] Implement bidirectional LSTM
- [ ] Add attention mechanisms
- [ ] Experiment with different embedding techniques (GloVe, Word2Vec)
- [ ] Add hyperparameter tuning
- [ ] Implement cross-validation
- [ ] Add model checkpointing and early stopping
