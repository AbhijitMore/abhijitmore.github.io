---
title: "Medicompanion RAG Application: Transforming Healthcare Information Delivery"
date: 2024-11-01
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

    MediCompanion
    ├── README.md               # Project documentation
    ├── main.py                 # Main application entry point
    ├── chat_model.py           # Core chat model implementation
    ├── config.py               # Configuration settings
    ├── document_loader.py      # Document loading and preprocessing
    ├── rag-datasets/           # Folder for datasets
    │   ├── gym supplements     # Gym supplements documents
    │   └── health supplements  # Health supplements documents
    ├── resources/              # Resources for output examples
    ├── retriever.py            # Document retrieval functions
    ├── text_splitter.py        # Splits text for efficient processing
    └── vector_store.py         # Embeddings storage for quick retrieval

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

- **Console Output**: Records of real-time chat outputs, providing insights into user interactions.

---

## 🤝 Contributions

Contributions are warmly welcomed! Whether you’re interested in enhancing features, refining responses, or adding new datasets, all efforts are appreciated. Fork the repository and submit a pull request to join in on advancing healthcare AI applications with Medicompanion.

---