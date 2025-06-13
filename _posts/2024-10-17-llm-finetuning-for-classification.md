--- 
title: "Fine-Tuning GPT-2 for SMS Spam Classification"
date: 2024-10-17
categories: [Projects, Gen AI] 
tags: [Deep Learning, LLM, FineTuning]

---

[![Open in Github Page](https://img.shields.io/badge/Hosted_with-GitHub_Pages-blue?logo=github&logoColor=white)](https://github.com/AbhijitMore/llm-finetuning-for-classification/)
<br>

# 📩 Fine-Tuning GPT-2 for SMS Spam Classification

The recent rise in spam messages has led to increased interest in leveraging advanced language models to counter the problem. This blog explores how to fine-tune a pre-trained GPT-2 model for SMS spam classification, covering the entire process of setting up, training, and evaluating a model to distinguish between spam and ham (non-spam) messages. Here, we’ll provide insights on the project structure, dataset, and visualized results.

## 🎯 Project Overview

This project fine-tunes GPT-2, a popular language model, to classify SMS messages effectively as spam or ham. The fine-tuning process builds on top of GPT-2’s language understanding, adapting it for classification. By the end, the model will be able to label new SMS messages based on the training it received from a well-known SMS spam dataset. 

---

## 📂 Directory Structure

Organizing files and code is essential for a streamlined project experience. Here’s a structured breakdown of our files:

    llm-finetuning-for-classification
    ├── README.md                 # Project documentation
    ├── main.py                   # Script for training, validation, and testing
    ├── review_classifier.pth     # Model checkpoint
    ├── requirements.txt          # Dependencies
    ├── sms_spam_collection/      # Dataset folder
    │   ├── SMSSpamCollection.tsv # SMS dataset file
    ├── classifier/               # Spam prediction utilities
    │   ├── predictor.py          # Spam prediction functions
    │   └── spam_classifier.py    # Classification helper functions
    ├── gpt2/                     # GPT-2 model files
    ├── nets/                     # Model architecture and configurations
    ├── resources/                # Results resources for visualization
    └── utils/                    # Utility scripts

### Key Files

- **main.py**: Core script that manages training, validation, and testing.
- **review_classifier.pth**: The saved model checkpoint, ready for reuse or deployment.
- **predictor.py**: Prediction utilities used to classify new messages.
- **resources/**: Folder containing accuracy and loss plots, visualizing the training process.

---

## 🔧 Getting Started

To begin, clone the repository and install the necessary packages.

```bash
git clone https://github.com/AbhijitMore/llm-finetuning-for-classification.git
cd llm-finetuning-for-classification
pip install -r requirements.txt
```

---

## 📊 The Dataset

This project uses the **SMS Spam Collection** dataset from the UCI Machine Learning Repository. It contains labeled SMS messages, divided into two columns:

- **Label**: Specifies if the message is spam or ham.
- **Text**: The actual SMS message content.

Once downloaded, place the dataset in the `sms_spam_collection/SMSSpamCollection.tsv` file path.

---

## 🚀 Model Training & Evaluation

### Training the Model

Run the `main.py` script to train the model on the SMS dataset:

```bash
python main.py
```

This script handles data loading, preprocessing, and fine-tuning of GPT-2 for classification. Loss and accuracy are computed, and graphs are generated throughout training.

### Testing & Results

After training, the model automatically tests on a designated dataset, with results outputted to the console. You’ll also find saved plots of accuracy and loss in the `resources/` folder:

- **Accuracy Plot** (`accuracy-plot.png`): Visual representation of model accuracy over epochs.
- **Loss Plot** (`loss-plot.png`): Indicates model loss progression, helping assess convergence.

### Console Output

During training, the console provides real-time updates on metrics like loss and accuracy, giving insights into the model’s performance as it improves over epochs.

---

## 📈 Results

Here’s a look at the final output:

- **Console Output**: Snapshots of training progression are saved in `resources/console_output.png`.
- **Performance Graphs**: The loss and accuracy plots summarize model stability and improvements throughout training.

---

## 🤝 Contributions

Contributions are welcomed! Whether suggesting a feature, reporting a bug, or submitting a pull request, all efforts are valued. Your input will help improve the SMS Spam Classification project and further empower GPT-2’s application to real-world classification tasks.

---