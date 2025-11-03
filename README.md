# Emotion and Empathy Prediction – Multi-Model Comparison

## Table of Contents
- [Task 1: Dataset and Key Attributes](#task-1-dataset-and-key-attributes)
  - [Dataset Overview](#dataset-overview)
  - [Data Preprocessing and Splits](#data-preprocessing-and-splits)
- [Task 2: Model Implementations](#task-2-model-implementations)
  - [Model 1: ANN with SentenceTransformer Embeddings](#model-1-ann-with-sentencetransformer-embeddings)
  - [Model 2: RNN with GloVe Embeddings](#model-2-rnn-with-glove-embeddings)
  - [Model 3: OpenAI Prompting (GPT-based Zero/Few-Shot)](#model-3-openai-prompting-gpt-based-zerofew-shot)
  - [Model 4: BERT Transformer Fine-Tuning](#model-4-bert-transformer-fine-tuning)
- [Task 3: Evaluation and Results](#task-3-evaluation-and-results)
  - [Classification Metrics](#classification-metrics)
  - [Regression Metrics](#regression-metrics)
  - [Confusion Matrices and Error Analysis](#confusion-matrices-and-error-analysis)
- [Task 4: Reproducibility Notes and Running Instructions](#task-4-reproducibility-notes-and-running-instructions)

---

# Task 1: Dataset and Key Attributes

## Dataset Overview
The dataset used for this project contains conversational text samples labeled for **three outputs**:
1. **Emotion (categorical)** – positive, neutral, negative  
2. **Emotion Intensity (regression)** – ordinal 0–5 scale  
3. **Empathy (regression)** – ordinal 0–5 scale  

Total samples: ~20k  
Split: 70% train, 15% validation, 15% test  
Source: TRAC2 / EmpatheticDialogues corpus (CSV formatted)  

Each record includes:
- `id`: unique identifier  
- `text`: input conversational utterance  
- `emotion_class`: categorical label  
- `emotion_intensity`: numeric  
- `empathy`: numeric  

### Data Preprocessing and Splits
- Lower-casing, punctuation and stopword removal using NLTK.  
- Tokenization handled differently per model (BERT tokenizer, GloVe tokenizer, etc.).  
- Stratified splits to preserve emotion distribution.  
- CSV files:
  - `trac2_CONVT_train.csv`
  - `trac2_CONVT_val.csv`
  - `trac2_CONVT_test.csv`

---

# Task 2: Model Implementations

## Model 1: ANN with SentenceTransformer Embeddings
**Implementation Details**
- Used `sentence-transformers/all-MiniLM-L6-v2` to convert sentences into 384-dimensional dense vectors.
- Each vector fed into a two-branch neural network:
  - **Classification head:** 2 fully connected layers with ReLU → Softmax (3 classes).
  - **Regression heads:** 2 dense layers each producing a single value for intensity and empathy.
- Optimizer: AdamW, LR=1e-4, batch size=64, dropout=0.3.
- Losses combined as weighted sum: `CrossEntropy + 0.5*(MSE_intensity + MSE_empathy)`.

**Results (Dev Set)**
- Accuracy: 0.72  
- F1 (macro): 0.70  
- MSE (Intensity): 0.40  
- MSE (Empathy): 0.76  

**Notes**
- Pre-computed embeddings cached to speed up training.
- Sensitive to learning rate — higher LR led to instability.

---

## Model 2: RNN with GloVe Embeddings
**Implementation Details**
- Tokenized text with Keras tokenizer, padded to 100 tokens.
- Initialized embedding matrix from `glove.6B.100d.txt`.
- Model architecture:
  - Embedding layer (frozen or fine-tuned)
  - Bidirectional LSTM(128) + Dropout(0.2)
  - Shared hidden layer → three outputs (1 classification, 2 regression).
- Optimizer: Adam (LR=1e-3), batch size=64, loss same as Model 1.

**Results (Dev Set)**
- Accuracy: 0.68  
- F1 (macro): 0.66  
- MSE (Intensity): 0.43  
- MSE (Empathy): 0.79  

**Notes**
- Slower training than ANN due to sequence processing.
- Captures sequential context but less robust on short texts.

---

## Model 3: OpenAI Prompting (GPT-based Zero/Few-Shot)
**Implementation Details**
- Used GPT-4 API to prompt directly on each utterance:
  - Example:  
    > “Given the text below, classify its emotional polarity (positive/neutral/negative),  
    > rate emotional intensity (0–5), and empathy (0–5). Respond as JSON.”
- Batched inference using Python `openai` client.
- No explicit training; relies on model understanding.

**Results (Dev Set)**
- Accuracy: 0.77  
- F1 (macro): 0.74  
- MSE (Intensity): 0.36  
- MSE (Empathy): 0.69  

**Notes**
- Few-shot examples improved consistency.
- Expensive for large datasets; best for evaluation or bootstrapping.

---

## Model 4: BERT Transformer Fine-Tuning
**Implementation Details**
- Base model: `bert-base-uncased` via Hugging Face Transformers.
- Multi-head architecture:
  - Shared BERT encoder.
  - One dense layer for classification logits (3 classes).
  - Two regression heads for intensity and empathy.
- Optimizer: AdamW (LR=2e-5), Scheduler: linear warmup.
- Fine-tuned for 4 epochs on GPU (40 GB VRAM).
- Used Trainer API with custom compute_metrics (F1, MSE).

**Results (Dev Set)**
- Accuracy: 0.73  
- F1 (macro): 0.71  
- MSE (Intensity): 0.40  
- MSE (Empathy): 0.76  

**Notes**
- Best generalization among trained models.
- Overfits after ~4 epochs — early stopping applied.

---

# Task 3: Evaluation and Results

## Classification Metrics
| Model | Accuracy | F1 (macro) |
|--------|-----------|------------|
| ANN (MiniLM) | 0.72 | 0.70 |
| RNN (GloVe) | 0.68 | 0.66 |
| GPT Prompting | **0.77** | **0.74** |
| BERT Fine-Tuned | 0.73 | 0.71 |

## Regression Metrics
| Model | MSE Intensity | MSE Empathy | MAE Intensity | MAE Empathy |
|--------|---------------|--------------|----------------|--------------|
| ANN (MiniLM) | 0.40 | 0.76 | 0.39 | 0.66 |
| RNN (GloVe) | 0.43 | 0.79 | 0.41 | 0.69 |
| GPT Prompting | **0.36** | **0.69** | **0.32** | **0.61** |
| BERT Fine-Tuned | 0.40 | 0.76 | 0.38 | 0.67 |

## Confusion Matrices and Error Analysis
- Most misclassifications occur between *neutral* and *positive* classes.
- BERT and GPT better handle ambiguous emotional tone.
- Regression errors are higher for empathy extremes (0 or 5) due to limited samples.

---

# Task 4: Reproducibility Notes and Running Instructions

## Environment
- Python 3.11  
- PyTorch 2.3  
- Transformers 4.44  
- SentenceTransformers 2.7  
- Pandas, Numpy, Scikit-learn, Matplotlib  

## How to Run
1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
