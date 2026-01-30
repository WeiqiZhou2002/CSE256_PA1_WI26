# CSE256 PA1: Sentiment Analysis with BOW and DAN Models

This project implements various neural network models for sentiment analysis, including Bag of Words (BOW) and Deep Averaging Networks (DAN). It involves training models on sentiment datasets and evaluating their performance.

## Project Structure

- `main.py`: The main entry point for training and evaluating models.
- `BOWmodels.py`: Implementation of Bag of Words models (`NN2BOW`, `NN3BOW`).
- `DANmodels.py`: Implementation of Deep Averaging Network models and associated datasets.
- `BPE.py`: Byte Pair Encoding implementation for subword tokenization.
- `sentiment_data.py`: Utilities for reading sentiment data and word embeddings.
- `utils.py`: General utility functions.
- `data/`: Directory containing datasets and embeddings.

## Setup & Requirements

Ensure you have Python 3 and the following dependencies installed:

```bash
pip install torch scikit-learn matplotlib
```

### Data Preparation
The project expects the following data files in the `data/` directory:
- `data/train.txt`: Training dataset
- `data/dev.txt`: Development/Validation dataset
- `data/glove.6B.50d-relativized.txt`: Pre-trained GloVe embeddings

## Usage

You can run the models using `main.py` with the `--model` argument.

### 1. Bag of Words (BOW)
Trains and evaluates 2-layer and 3-layer Feed Forward Neural Networks using Bag of Words features.

```bash
python main.py --model BOW
```
Output:
- Generates `train_accuracy.png` and `dev_accuracy.png`.

### 2. DAN with Random Embeddings
Trains a Deep Averaging Network (DAN) using randomly initialized word embeddings.

```bash
python main.py --model DAN_RANDOM
```
Output:
- Generates `dan_random_accuracy.png`.

### 3. DAN with GloVe Embeddings
Trains a DAN using pre-trained GloVe embeddings.

```bash
python main.py --model DAN
```
Output:
- Generates `dan_accuracy.png`.

### 4. Subword DAN (BPE)
Trains a Byte Pair Encoding (BPE) tokenizer and then trains a DAN model using subword embeddings.

```bash
python main.py --model SUBWORDDAN
```
Output:
- Generates `subword_dan_accuracy.png`.

## Outputs
The script prints training and validation accuracy for each epoch to the console and saves plots of the accuracy curves in the current directory.
