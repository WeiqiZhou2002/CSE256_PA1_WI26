# models.py

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from sentiment_data import read_sentiment_examples, read_word_embeddings
from torch.utils.data import Dataset, DataLoader
import time
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from BOWmodels import SentimentDatasetBOW, NN2BOW, NN3BOW
from DANmodels import DAN, SentimentDatasetDAN, SentimentDatasetBPE, SubwordEmbeddings, RandomEmbeddings
from BPE import BPE


# Training function
# Replace these functions in main.py

def train_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(data_loader):
        if X.dtype != torch.long:
            X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_train_loss = train_loss / num_batches
    accuracy = correct / size
    return accuracy, average_train_loss


def eval_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    eval_loss = 0
    correct = 0
    for batch, (X, y) in enumerate(data_loader):
        # FIX: Same check for evaluation
        if X.dtype != torch.long:
            X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        eval_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    average_eval_loss = eval_loss / num_batches
    accuracy = correct / size
    return accuracy, average_eval_loss

# Experiment function to run training and evaluation for multiple epochs
def experiment(model, train_loader, test_loader):
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    all_train_accuracy = []
    all_test_accuracy = []
    for epoch in range(100):
        train_accuracy, train_loss = train_epoch(train_loader, model, loss_fn, optimizer)
        all_train_accuracy.append(train_accuracy)

        test_accuracy, test_loss = eval_epoch(test_loader, model, loss_fn, optimizer)
        all_test_accuracy.append(test_accuracy)

        if epoch % 10 == 9:
            print(f'Epoch #{epoch + 1}: train accuracy {train_accuracy:.3f}, dev accuracy {test_accuracy:.3f}')
    
    return all_train_accuracy, all_test_accuracy


def main():

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run model training based on specified model type')
    parser.add_argument('--model', type=str, required=True, help='Model type to train (e.g., BOW)')

    # Parse the command-line arguments
    args = parser.parse_args()

    start_time = time.time()
    train_data = SentimentDatasetBOW("data/train.txt")
    dev_data = SentimentDatasetBOW("data/dev.txt", vectorizer=train_data.vectorizer, train=False)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(dev_data, batch_size=16, shuffle=False)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Data loaded in : {elapsed_time} seconds")


    if args.model == "BOW":
        start_time = time.time()
        print('\n2 layers:')
        nn2_train_accuracy, nn2_test_accuracy = experiment(NN2BOW(input_size=512, hidden_size=100), train_loader, test_loader)

        print('\n3 layers:')
        nn3_train_accuracy, nn3_test_accuracy = experiment(NN3BOW(input_size=512, hidden_size=100), train_loader, test_loader)

        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_train_accuracy, label='2 layers')
        plt.plot(nn3_train_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for 2, 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        training_accuracy_file = 'train_accuracy.png'
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_test_accuracy, label='2 layers')
        plt.plot(nn3_test_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for 2 and 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the testing accuracy figure
        testing_accuracy_file = 'dev_accuracy.png'
        plt.savefig(testing_accuracy_file)
        print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")

        # plt.show()
    elif args.model == "DAN_RANDOM":
        print("Running Part 1b: Randomly Initialized Embeddings")

        print("Loading vocabulary...")
        glove_embeddings = read_word_embeddings("data/glove.6B.50d-relativized.txt")
        vocab_size = len(glove_embeddings.word_indexer)

        random_embeddings = RandomEmbeddings(vocab_size=vocab_size, embedding_dim=50)

        train_data = SentimentDatasetDAN("data/train.txt", glove_embeddings)
        dev_data = SentimentDatasetDAN("data/dev.txt", glove_embeddings)

        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        test_loader = DataLoader(dev_data, batch_size=16, shuffle=False)

        print('\nTraining DAN (Random Init):')
        model = DAN(random_embeddings, hidden_size=100)

        rand_train_acc, rand_dev_acc = experiment(model, train_loader, test_loader)

        plt.figure(figsize=(8, 6))
        plt.plot(rand_train_acc, label='Train')
        plt.plot(rand_dev_acc, label='Dev')
        plt.title('DAN Accuracy (Random Initialization)')
        plt.legend()
        plt.grid()
        plt.savefig('dan_random_accuracy.png')
        print("Plot saved to dan_random_accuracy.png")

    elif args.model == "DAN":
        start_time = time.time()
        print("Loading DAN Data and Embeddings...")

        word_embeddings = read_word_embeddings("data/glove.6B.50d-relativized.txt")

        train_data = SentimentDatasetDAN("data/train.txt", word_embeddings)
        dev_data = SentimentDatasetDAN("data/dev.txt", word_embeddings)

        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        test_loader = DataLoader(dev_data, batch_size=16, shuffle=False)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Data loaded in : {elapsed_time} seconds")

        print('\nTraining DAN:')
        model = DAN(word_embeddings, hidden_size=100)

        dan_train_acc, dan_dev_acc = experiment(model, train_loader, test_loader)

        plt.figure(figsize=(8, 6))
        plt.plot(dan_train_acc, label='Train')
        plt.plot(dan_dev_acc, label='Dev')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('DAN Accuracy')
        plt.legend()
        plt.grid()
        plt.savefig('dan_accuracy.png')
        print("DAN accuracy plot saved as dan_accuracy.png")

    elif args.model == "SUBWORDDAN":
        print("Training BPE...")
        vocab_size = 2000
        bpe = BPE(vocab_size=vocab_size)
        bpe.train("data/train.txt")

        print("Loading BPE Data...")
        train_data = SentimentDatasetBPE("data/train.txt", bpe)
        dev_data = SentimentDatasetBPE("data/dev.txt", bpe)

        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        test_loader = DataLoader(dev_data, batch_size=16, shuffle=False)

        subword_embeddings = SubwordEmbeddings(vocab_size=len(bpe.token_to_idx), embedding_dim=50)

        print('\nTraining Subword DAN:')
        model = DAN(subword_embeddings, hidden_size=100)

        subword_train_acc, subword_dev_acc = experiment(model, train_loader, test_loader)

        plt.figure(figsize=(8, 6))
        plt.plot(subword_train_acc, label='Train')
        plt.plot(subword_dev_acc, label='Dev')
        plt.title(f'Subword DAN Accuracy (Vocab {vocab_size})')
        plt.legend()
        plt.grid()
        plt.savefig('subword_dan_accuracy.png')

if __name__ == "__main__":
    main()
