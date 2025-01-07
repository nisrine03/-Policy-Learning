# Policy Learning for Dynamic Feature Selection in Text Classification using JAX

## Overview
This project leverages reinforcement learning (RL) to dynamically select the most informative features from textual data to enhance the efficiency and accuracy of text classification. By using the JAX library for high-performance computations, the project demonstrates an innovative approach to feature selection and model optimization.

### Dataset Description
The dataset consists of **21,958 news articles**, each labeled with one of **10 distinct categories**. However, the dataset presents challenges like class imbalance and heterogeneous sequence lengths, which are addressed in detail in this project.

---

## Key Steps in the Project

### 1. Class Analysis

#### 1.1 Class Imbalance
- **Problem**: The dataset exhibits class imbalance, with some categories being significantly overrepresented.
- **Solution**:  
  - Truncated the majority classes to 500 samples each to achieve balance.
  - Applied **upsampling** using **NLPaug** for generating diverse textual data. This helps improve the performance of NLP models by augmenting the dataset (e.g., `ContextualWordEmbsAug`).

#### 1.2 Sequence Length Analysis
- **Observation**: Sequence lengths vary greatly, indicating a heterogeneous distribution.
- **Problem**: Extreme variability in sequence length may degrade model performance if embeddings fail to capture the underlying information effectively.
- **Solution**: Selected an embedding method (BERT) capable of handling both long and short sequences without significant information loss.

---

### 2. Data Preprocessing

#### 2.1 Text Cleaning
- Removed special characters.
- Converted text to lowercase.
- Removed accents.
- Filtered out short words.
- Reduced multiple spaces to single spaces.
- Eliminated stopwords.

#### 2.2 Text Embeddings with BERT
- **Tokenizer**: Used `bert-base-uncased` tokenizer to convert text into sequences suitable for BERT processing, including padding and truncation.
- **Embedding Extraction**: Generated contextual embeddings for each token, representing words within their context.
- **Sentence Pooling**: Saved tokens and embeddings to ensure reproducibility and optimize future computations.

---

### 3. Limitations of Traditional Feature Selection
- Methods like **TF-IDF** struggle to capture the contextual and sequential nature of textual data, making them less effective for dynamic feature selection in this project.

---

### 4. Bidirectional LSTMs with Attention
- **Model**: Implemented a Bidirectional LSTM model with an attention mechanism.
  - Capable of processing sequential data.
  - Allows the model to focus on the most relevant parts of the input sequence.
- **Purpose**: Used as a baseline for comparison and as input for the RL agent and environment.

---

### 5. RL Environment for Dynamic Feature Selection

The RL environment is designed to facilitate the dynamic selection of text features, optimizing predictive performance and feature sparsity.

#### Key Objectives of the RL Environment:
1. **Dynamic Interaction**: Ensure effective interaction between the RL agent and the textual data.
2. **Performance Optimization**: Improve classification accuracy while minimizing computational costs.
3. **Modeling Flexibility**: Adapt to different feature selection strategies and datasets.

#### RL Workflow:
- The environment guides the agent to:
  - Select the most relevant text features.
  - Maximize model performance.
  - Minimize computational costs and feature redundancy.

---

## Tools and Technologies
- **JAX**: High-performance numerical computation library for RL implementations.
- **NLPaug**: Data augmentation library for NLP tasks.
- **BERT**: Pre-trained transformer model for generating contextual embeddings.
- **Bidirectional LSTM**: Sequential model with an attention mechanism for feature importance analysis.

---

## Key Contributions
1. Demonstrates how reinforcement learning can optimize feature selection dynamically in text classification.
2. Explores the integration of BERT embeddings into an RL framework.
3. Provides solutions to common challenges in text classification, such as class imbalance and sequence length variability.

---

## Installation and Usage

### Prerequisites
- Python 3.8+
- JAX, NumPy, PyTorch, and other dependencies listed in `requirements.txt`.

### Steps to Run the Project
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/dynamic-feature-selection-text.git
   cd dynamic-feature-selection-text
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Preprocess the data:
   ```bash
   python preprocess_data.py
   ```
4. Train the Bidirectional LSTM model:
   ```bash
   python train_baseline_model.py
   ```
5. Run the RL agent:
   ```bash
   python train_rl_agent.py
   ```

---

## Future Work
- Extend the RL environment to handle multilingual datasets.
- Integrate other advanced embedding models, such as GPT or RoBERTa.
- Evaluate performance on real-world text classification tasks like sentiment analysis and topic detection.

---

## License
This project is licensed under the MIT License. See `LICENSE` for more details.

---

## Acknowledgments
- Special thanks to the developers of JAX and NLPaug for their powerful libraries.
- Inspired by state-of-the-art research in text classification and feature selection.


