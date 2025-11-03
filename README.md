# Emotion and Empathy Prediction – Multi-Model Comparison

## Table of Contents
- [Task 1: Dataset and Key Attributes](#task-1-dataset-and-key-attributes)
  - [Dataset Overview](#dataset-overview)
  - [Data Preprocessing and Splits](#data-preprocessing-and-splits)
- [Task 2: Model Implementations](#task-2-model-implementations)
  - [Model 1: ANN with SentenceTransformer Embeddings](#model-1-ann-with-sentencetransformer-embeddings)
  - [Model 2: RNN with GloVe Embeddings](#model-2-rnn-with-glove-embeddings)
  - [Model 3: BERT Transformer Fine-Tuning](#model-3-bert-transformer-fine-tuning)
  - [Model 4: OpenAI Prompting (GPT-based Zero/Few-Shot)](#model-4-openai-prompting-gpt-based-zerofew-shot)
- [Task 3: Evaluation and Results](#task-3-evaluation-and-results)
  - [Classification Metrics](#classification-metrics)
  - [Regression Metrics](#regression-metrics)
  - [Confusion Matrices and Error Analysis](#confusion-matrices-and-error-analysis)
- [Task 4: Reproducibility Notes and Running Instructions](#task-4-reproducibility-notes-and-running-instructions)

---

# Task 1: Dataset and Key Attributes

## Dataset Overview
The dataset used for this project contains conversational text samples labeled for **three outputs**:
1. **Emotional Polarity (categorical)** – positive, neutral, negative  
2. **Emotion Intensity (regression)** – ordinal 0–5 scale  
3. **Empathy (regression)** – ordinal 0–5 scale  

Total samples: ~20k  
Source: TRAC2 / EmpatheticDialogues corpus (CSV formatted)  

Each record includes:
- `id`: unique identifier  
- `text`: input conversational utterance  
- `emotion_class`: categorical label  
- `emotion_intensity`: numeric  
- `empathy`: numeric  

- CSV files:
  - `trac2_CONVT_train.csv`
  - `trac2_CONVT_dev.csv`
  - `trac2_CONVT_test.csv`

### Data Preprocessing and Splits
- Lower-casing, punctuation and stopword removal using NLTK.  
- Tokenization handled differently per model (BERT tokenizer, GloVe tokenizer, etc.).   

---

# Task 2: Model Implementations

## Model 1: ANN with SentenceTransformer Embeddings
**Implementation Details**
- Used `sentence-transformers/all-MiniLM-L6-v2` to convert sentences into 384-dimensional dense vectors
- Each vector fed into a two-branch neural network:
  - **Classification head:** 2 fully connected layers with ReLU → Softmax 
  - **Regression heads:** 2 fully connected layers with ReLU
- Optimizer: AdamW, LR=1e-3, batch size=32, dropout=0.2, epoch=30

**Results (Dev Set)**
- Accuracy: 0.655
- F1: 0.649  
- MSE (Intensity): 0.42
- MSE (Empathy): 0.84

---

## Model 2: RNN with GloVe Embeddings
**Implementation Details**
- Tokenized text and padding to 128 tokens per sentence
- Initialized embedding matrix from `glove.6B.100d.txt`.
- Model architecture:
  - Embedding layer 
  - Bidirectional LSTM(128) + Dropout(0.2)
  - Shared hidden layer → three outputs (1 classification, 2 regression).
- Optimizer: Adam (LR=1e-3), batch size=32, epoch=40

**Results (Dev Set)**
- Accuracy: 0.599  
- F1 (macro): 0.596  
- MSE (Intensity): 0.55 
- MSE (Empathy): 1.14

---


## Model 3: BERT Transformer Fine-Tuning
**Implementation Details**
- Base model: `bert-base-uncased` via Hugging Face Transformers.
- Multi-head architecture:
  - Shared BERT encoder
  - One dense layer for classification logits 
  - Two regression heads for intensity and empathy.
- Optimizer: AdamW (LR=2e-5), Scheduler: linear warmup, weight decay = 0.01
- Fine-tuned for 4 epochs on GPU A100 (40 GB VRAM).
- Used Trainer API with custom compute_metrics (F1, MSE)

**Results (Dev Set)**
- Accuracy: 0.725
- F1: 0.718
- MSE (Intensity): 0.40  
- MSE (Empathy): 0.76  

---

## Model 4: OpenAI Prompting (GPT-based Zero/Few-Shot)
**Implementation Details**
- Used GPT-5 API to prompt directly on each utterance:
  - Example:  
    > “Given the text below, classify its emotional polarity (positive/neutral/negative),  
    > rate emotional intensity (0–5), and empathy (0–5). Respond as JSON.”
- Batched inference using Python `openai` client.
- No explicit training; relies on model understanding.
- Only use 500 samples in training set to train and 100 samples in dev set to test

**Results (Dev Set)**
- Accuracy: 0.770  
- F1: 0.779
- MSE (Intensity): 0.58
- MSE (Empathy): 1.02

---

# Task 3: Evaluation and Results

## Classification Metrics 
| Model | Accuracy | F1 |
|--------|-----------|------------|
| Base Guessing | 0.333 | 0.333 |
| ANN + ST | 0.655 | 0.649 |
| RNN (GloVe) | 0.599 | 0.596 |
| BERT Fine-Tuned | 0.725 | 0.718 |
| GPT Prompting | 0.770 | 0.779 |

## Regression Metrics
| Model | MSE Intensity | MSE Empathy |
|--------|---------------|--------------|
| ANN + ST | 0.42 | 0.84 |
| RNN (GloVe) | 0.55 | 1.14 |
| BERT Fine-Tuned | 0.40 | 0.76 |
| GPT Prompting | 0.58 | 1.02 |

---

# Task 4: Reproducibility Notes and Running Instructions

## Environment
- Python 3.11  
- PyTorch 2.3  
- Transformers 4.44  
- SentenceTransformers 2.7  
- Pandas, Numpy, Scikit-learn, Matplotlib  

## How to Run
To reproduce the results, the project can be executed either on **Google Colab** or on a **local machine**. The recommended approach is Google Colab, as it provides a preconfigured environment with GPU support and minimal setup effort. Users simply open the Colab notebook, switch the runtime to GPU under “Runtime > Change runtime type > Hardware accelerator,” and verify the GPU using `torch.cuda.get_device_name(0)`. After cloning or uploading the project repository, install dependencies with `!pip install -r requirements.txt`

For local execution, I recommend installing **pyenv** to manage Python versions and create a clean environment. Once Python 3.11 is installed through pyenv, set up a virtual environment and run `pip install -r requirements.txt` to install all dependencies. Training and evaluation commands remain identical to those used on Colab. 