# The Next Clue: Predicting Words with LSTMs

Can a neural network learn enough from a classic novel to suggest the word that comes next? This project explores that question by building a next word prediction model with TensorFlow and Keras, using the text of Arthur Conan Doyle's *The Adventures of Sherlock Holmes*.

The notebook takes the reader through the full experiment, from preparing raw text and creating training sequences to building LSTM models, monitoring their performance and testing a prediction. It also compares a baseline model with a larger version that uses dropout and early stopping.

## Project Overview

The goal is to train a model that receives a sequence of five words and predicts the most likely sixth word. The project is a practical introduction to natural language processing, tokenization, word embeddings and recurrent neural networks.

## Dataset

The text comes from Project Gutenberg's eBook **The Adventures of Sherlock Holmes** by Arthur Conan Doyle, eBook number 1661.

The notebook loads a text file named `book.txt` from Google Drive. It converts the text to lowercase and uses the first 20,000 whitespace separated tokens for the experiment. Keras Tokenizer then converts the text into integer word identifiers.

The resulting vocabulary contains 3,535 entries. The sequence preparation step produces 16,086 usable examples, each containing five input words and one target word.

## Tools and Technologies

1. Python for data preparation and model development.
2. TensorFlow and Keras for tokenization, model building and training.
3. NumPy for preparing the input and target arrays.
4. Matplotlib for plotting training and validation accuracy and loss.
5. Google Colab and Google Drive for running the notebook and loading the text file.

## Methodology

### 1. Text Preparation

The notebook reads the source text, converts it to lowercase and limits the experiment to the first 20,000 whitespace separated tokens. A Keras Tokenizer builds a word index and maps the text to integer sequences.

### 2. Sequence Generation

A sliding window creates examples from the tokenized text. Each example uses five consecutive words as input and the following word as its target. This turns the original text into a supervised learning problem.

The input array has 16,086 rows and five columns. The target word is represented using one hot encoding across the 3,535 vocabulary entries.

### 3. Baseline LSTM Model

The first model uses an embedding layer with 64 dimensions, an LSTM layer with 128 units, a dense layer with 128 neurons and a softmax output layer. It is compiled with the Adam optimizer and categorical cross entropy loss, then trained for 10 epochs with a batch size of 128 and a 10 percent validation split.

### 4. Larger Model with Regularization

The notebook then tests a larger architecture with a 128 dimensional embedding, an LSTM layer with 256 units, a dropout layer with a rate of 0.3, a dense layer with 256 neurons and a softmax output layer.

This version uses the same optimizer and loss function. It is configured for up to 30 epochs and uses early stopping to monitor validation loss. Training stops when validation loss does not improve for three consecutive epochs, and the callback restores the weights from the best validation loss epoch.

### 5. Prediction and Training Diagnostics

A prediction function converts a supplied text prompt into token IDs, pads or truncates it to the five word context length and passes it to the trained model. The word with the highest predicted probability is returned.

Matplotlib is used to compare training and validation accuracy and loss across epochs. These plots help show how well the model learns the training text and whether its performance carries over to the validation portion.

## Results

### Baseline Model

At the end of its 10 training epochs, the baseline model recorded:

1. Training accuracy: **8.95 percent**.
2. Validation accuracy: **7.21 percent**.
3. Training loss: **5.2680**.
4. Validation loss: **7.1868**.

### Larger Model with Early Stopping

The larger model stopped after five epochs because its validation loss did not improve for three consecutive epochs. Its best recorded validation loss was **6.4507** at epoch two. At epoch five, the training log showed 7.53 percent training accuracy and 5.97 percent validation accuracy. The early stopping callback was configured to restore the weights from the best validation loss epoch.

The notebook does not include a separate evaluation after those weights are restored, so the epoch five accuracy should not be treated as the final score of the restored model.

### Example Prediction

The notebook tests the prompt:

```text
sherlock holmes was sitting in
```

The model predicts:

```text
the
```

This is an example of the prediction function working on one prompt, rather than evidence of general language understanding or strong predictive performance.

## What This Project Explores

1. Turning unstructured text into sequences that can be used for supervised learning.
2. Building a word index and embedding layer for a language modelling task.
3. Using an LSTM network to learn patterns across a short sequence of words.
4. Comparing a baseline architecture with a larger model that includes dropout.
5. Using validation loss and early stopping to monitor training.
6. Visualizing model performance and testing next word predictions.

## Limitations and Possible Improvements

This is an exploratory project rather than a production ready text prediction system. The experiment uses only the first 20,000 tokens from one novel, so the training data is limited. It uses a fixed context of five words and the notebook does not include a separate held out test set. The recorded validation accuracy is also modest, which indicates that the models have not learned to predict the next word reliably.

A useful next step would be to clean the Project Gutenberg header and footer, experiment with a larger text corpus, compare different context lengths and model sizes, and evaluate the final model on a separate test set. It would also be useful to compare top one accuracy with top five accuracy and inspect several predictions rather than relying on a single example.

## How to Run the Notebook

The notebook was prepared for Google Colab and reads `book.txt` from Google Drive.

1. Upload the Project Gutenberg text file to your Google Drive.
2. Open the notebook in Google Colab and run the Drive mounting cell.
3. Update `file_path` in the notebook so that it points to the location of your `book.txt` file.
4. Run the notebook cells in order to prepare the text, train the models, view the learning curves and try a next word prediction.

The main dependencies are TensorFlow, NumPy and Matplotlib. They can be installed in a Python environment with:

```bash
pip install tensorflow numpy matplotlib
```

## Source

Arthur Conan Doyle, *The Adventures of Sherlock Holmes*, Project Gutenberg eBook number 1661.

Project Gutenberg: https://www.gutenberg.org/ebooks/1661
