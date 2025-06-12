---
title: "Medicompanion RAG Application: Transforming Healthcare Information Delivery"
date: 2024-10-30
categories: [Projects, Deep Learning]
tags: [RAG, LangChain, LLaMA]

---

[![Open in Github Page](https://img.shields.io/badge/Hosted_with-GitHub_Pages-blue?logo=github&logoColor=white)](https://github.com/AbhijitMore/MediCompanion)
<br>

# 🤖💊 Medicompanion RAG Application: Transforming Healthcare Information Delivery

In today’s healthcare landscape, accurate information is paramount. Medicompanion is a cutting-edge Retrieval-Augmented Generation (RAG) application designed to deliver reliable health and fitness supplement information. This blog details the project setup, architecture, and the seamless user experience Medicompanion offers through advanced conversational AI built with LangChain and the LLaMA 3.2 model.

## 🎯 Project Overview

Medicompanion utilizes RAG and LLaMA 3.2 to answer health-related questions by retrieving relevant documents and generating conversational responses, ensuring clarity and relevance. With the focus on gym and health supplements, it’s an ideal tool for those looking to make informed decisions regarding their wellness routines.

---

## 📂 Directory Structure

A well-organized directory structure is key to managing the project efficiently. Below is the breakdown of Medicompanion’s directory:

```
.
├── README.md               # Documentation for the project
├── chat_model.py           # Implementation of the chat model
├── config.py               # Configuration settings for the application
├── document_loader.py      # Functions to load documents from the dataset
├── logger_config.py        # Logger configurations for the application
├── main.py                 # Entry point for the application
├── tasks.py                # all the logic for different tasks
├── rag-datasets            # Directory containing the datasets
│   ├── gym supplements     # Documents related to gym supplements
│   │   ├── 1. Analysis of Actual Fitness Supplement.pdf
│   │   └── 2. High Prevalence of Supplement Intake.pdf
│   └── health supplements  # Documents related to health supplements
│       ├── 1. dietary supplements - for whom.pdf
│       ├── 2. Nutraceuticals research.pdf
│       └── 3.health_supplements_side_effects.pdf
├── resources               # Additional resources (e.g., images, outputs)
│   └── console_output.png  # Example output screenshot
├── logs                    # Directory to store logs
│   └── application.log     # log file
├── retriever.py            # Functions to retrieve relevant documents
├── text_splitter.py        # Functions to split text for processing
└── vector_store.py         # Vector store implementation for embeddings
```

### Key Files

- **main.py**: Launches the Medicompanion chatbot application.
- **chat_model.py**: Implements the LLaMA model, enabling natural, responsive interactions.
- **document_loader.py**: Loads and processes health-related documents.
- **resources/**: Contains example outputs and images for easy reference.

---

## 🔧 Getting Started

To set up the Medicompanion application, clone the repository and install the necessary dependencies.

```bash
git clone https://github.com/AbhijitMore/MediCompanion.git
cd MediCompanion
pip install -U langchain-community faiss-cpu langchain-huggingface pymupdf langchain-ollama python-dotenv
```

After installing dependencies, start the chatbot application by running:

```bash
python main.py
```

This will initialize the Medicompanion chatbot, allowing users to ask questions related to health supplements and receive informative responses.

---

## 📊 Datasets

Medicompanion leverages categorized documents located in `rag-datasets/`. These datasets, covering gym and health supplements, help the chatbot provide reliable, accurate responses by storing embeddings that facilitate quick retrieval during conversations.

---

## 🚀 Application Performance & Features

### Interactive Responses

Once started, the Medicompanion application provides real-time responses based on user queries, drawing from its database of health and supplement documents. Responses can include side effects, benefits, and usage guidelines for various supplements.

---

## 📈 Results & Performance

Throughout its operation, Medicompanion outputs relevant response:

- **Console Output**: 
![user interaction](assets/img/console_output.png)

---

## 🤝 Contributions

Contributions are warmly welcomed! Whether you’re interested in enhancing features, refining responses, or adding new datasets, all efforts are appreciated. Fork the repository and submit a pull request to join in on advancing healthcare AI applications with Medicompanion.

---