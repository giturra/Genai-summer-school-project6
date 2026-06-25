# Project 6 — Sentiment Classification on Sentiment140

**International Summer School on Generative AI**  
**From NLP to LLMs to Agentic AI: Theory, Applications, and Practice**

**Integrants:**

- Patrizia De Camillis
- Bartas Lisauskas
- Mark Byrd
- Minghzi Wang
- Gabriel Iturra Bocaz

## Overview

This project compares four generations of NLP sentiment classifiers on the same binary tweet-classification task. The goal is to understand what each stage of NLP buys us in practice: a fast classical machine learning baseline, a small neural network, and a fine-tuned transformer model.

The task uses the **Sentiment140** dataset, which contains 1.6 million English tweets labeled as negative or positive. 

Because the full dataset is large, all three models were trained and evaluated on the same balanced **50,000-tweet subsample**:

- 25,000 negative tweets
- 25,000 positive tweets
- 40,000 training examples
- 10,000 test examples

## Dataset and Preprocessing

The dataset was loaded from Hugging Face using:

```python
load_dataset("contemmcm/sentiment140")
```

The dataset split used was:

```python
ds["complete"]
```

A fixed random seed was used to make the experiment reproducible.

Light tweet cleanup was applied before training:

- URLs were removed.
- User mentions such as `@johnsmith` were replaced with `@user`.
- Hashtags were left unchanged because they often carry sentiment.
- Extra whitespace was normalized.

Example cleanup:

```text
Before:  @johnsmith I love this! http://example.com #happy
After:   @user I love this! #happy
```

## Models

### Stage 1 — Classical Machine Learning

The classical baseline used a TF-IDF representation with Logistic Regression.

Features:

- Word-level TF-IDF features
- Character-level TF-IDF features
- Logistic Regression classifier

This model is fast, compact, and surprisingly strong for short noisy tweets.

### Stage 2 — Neural Model

The neural model used PyTorch with a trainable embedding layer and a bidirectional LSTM.

Architecture:

- Tokenizer built from the training data
- Trainable embedding layer
- Bidirectional LSTM
- Dropout
- Linear classification layer

This model learns dense word representations from scratch, rather than relying on sparse TF-IDF features.

### Stage 3 — BERT

The transformer model fine-tuned `distilbert-base-uncased` using the Hugging Face `Trainer` API.

Settings:

- Model: `distilbert-base-uncased`
- Epochs: 2
- Maximum sequence length: 64 tokens
- Binary sequence classification head

Tweets are usually short, so truncating to 64 tokens is enough for this task.

## Stage 4 - Large Language Model

We also test on zero-shot performance on the LLM `qwen-2.5-3b`. We sample the test set to 1000 (statistically permissable with a sampling error of ~3%) for more faster testing.

We test on two prompts (naive + detailed) and record the best result.

#

## Results

All models were trained and evaluated on the same 50,000-example balanced dataset.

| Model | Dataset Size | Train Size | Test Size | Accuracy | F1 | Train Time (s) | Inference (ms/example) | Model Size (MB) | Max Tokens | Epochs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 50,000 | 40,000 | 10,000 | 0.7968 | 0.798333 | 22.441771 | 0.253406 | 5.638535 | — | — |
| Embedding + BiLSTM | 50,000 | 40,000 | 10,000 | 0.7708 | 0.765019 | 11.808705 | 0.000414 | 6.084473 | 50 | 3 |
| DistilBERT fine-tune | 50,000 | 40,000 | 10,000 | 0.8295 | 0.825218 | 283.382543 | 7.116618 | 256.109494 | 64 | 2 |
|LLM| 10,000 | 0 | 10,000 | 0.7350 | 0.7271 | 0 | 543.54 | 2000 | 32,768 | 0 |


## Data Scaling

Since we only used a sample of 50,000 from the full dataset, we were left with the question of whether the models would scale. We realised that it was impossible to train the larger models on all 1.6 million data points, and so we decided to measure the scaling using the logistic regression model. This demonstrates that the accuracy scales with the data, as well as showing exactly how much more data efficient the pre-trained transformer is, in the large data regime.

We also showed that the pre-trained transformer is beaten by the linear model in the small data regime. 

## Abalations

We had many hypothesis as to why the BERT model performed the best; was it the attention mechanism, the bidirectionality, the pre-training or the model size? We managed to ablate most of these concerns, and were left with the conclusion that the pre-training is the most important part, but that model size is required to capitalise on it.

We did this by performing the exact same experiment on different models, to ablate bidirectionality we tested on a unidirectional LSTM, and achieved the same results. This implies that bidirectional consideration was not a key factor. We ablated the rest in a similar manner, and the details, along with comments of our thought process, are in the notebooks.

## Interpretation

The classical TF-IDF + Logistic Regression model performed very well, reaching almost 80% accuracy with a small model size and fast training. This shows that classical NLP methods are still strong baselines, especially for short texts like tweets.

The BiLSTM model performed slightly worse than the classical model in this run. This is not unusual. The model was trained from scratch on only 50,000 tweets, so it did not benefit from large pretrained language representations. Neural models often need more data, careful tuning, or pretrained embeddings such as GloVe or fastText to beat a strong TF-IDF baseline.

DistilBERT achieved the best accuracy and F1 score. This shows the advantage of transformer-based language models: they start with useful pretrained language knowledge and can better handle context, negation, and short phrase-level meaning. However, this improvement came with much higher training time, inference latency, and disk size.

The zero-shot LLM performed the worst, and the detailed prompt lowered the accuracy even more. We suspect this is because of the lack of fine-tuning, the LLM is not trained for twitter specific sentiment signals.

## Main Takeaways

1. **Classical NLP is still competitive.**  
   TF-IDF + Logistic Regression was fast, small, and accurate.

2. **Small neural models are not automatically better.**  
   The BiLSTM learned from scratch and did not outperform the classical baseline on this subsample.

3. **Transformers give the best performance but cost more.**  
   DistilBERT achieved the highest accuracy and F1, but required much more compute and storage.

4. **The Sentiment140 labels are noisy.**  
   Since labels were originally derived from emoticons, some examples are ambiguous or mislabeled. This limits maximum performance.

5. **The experiment reflects the evolution from NLP to LLMs.**  
   The project demonstrates the progression from sparse feature-based NLP, to neural representation learning, to pretrained transformer models, which are the foundation of modern LLM systems.

## Reproducibility

All experiments used:

- Dataset: Sentiment140
- Subsample size: 50,000 balanced tweets
- Train/test split: 80/20
- Random seed: 42
- Same cleaning strategy across all stages

## Running Interface

The project includes a local demo interface called **Y**, a mock social media
page where the three sentiment models reply to a posted tweet as comments. This
connects the notebook experiments to the presentation demo: the classical model,
the BiLSTM model, and the DistilBERT model all read the same input and return
their predicted sentiment.

First install the project dependencies:

```bash
pip install -r requirements.txt
```

Create a `models/` folder in the project root with the trained artifacts:

```text
models/
- classical.pkl
- bilstm.pt
- distilbert/
```

The files should contain the Logistic Regression pipeline, the trained BiLSTM
checkpoint, and the saved DistilBERT model directory respectively.

Run the demo with:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5057
```

The app runs locally with Flask. It loads the model artifacts once at startup,
then sends each tweet to the models through the `/predict` endpoint.

## Running the Notebooks

The notebooks are stored in the `notebooks/` folder and follow the same
experimental structure as the presentation:

1. Classical TF-IDF + Logistic Regression
2. Embedding + BiLSTM
3. Fine-tuned DistilBERT
4. LLM
5. Data Scaling
6. Ablation

## Project Context

This project was completed in the context of the **International Summer School on Generative AI: From NLP to LLMs to Agentic AI — Theory, Applications, and Practice**.

It connects directly to the course theme by showing how sentiment classification changes across four stages of NLP development:

1. Classical NLP with handcrafted sparse text features
2. Neural NLP with learned embeddings and recurrent networks
3. Fine-tuned BERT NLP with pretrained language models
4. Zero shot Large Language Models

This comparison helps explain why modern generative AI and agentic AI systems are built on transformer-based architectures, while also showing that simpler methods remain useful for many practical tasks. In addition, despite LLMs offered a high level of semantic understanding they are still sub-optimal for specific sentiment analysis tasks, and require additional fine-tuning.
