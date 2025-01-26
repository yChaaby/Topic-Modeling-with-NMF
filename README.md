# Topic Modeling using Non-Negative Matrix Factorization (NMF)

This project demonstrates the application of **Non-Negative Matrix Factorization (NMF)** for topic modeling on a dataset of abstracts. Below, you will find a detailed explanation of the dataset, target variable, mathematical background of NMF, and evaluation methodology.

```python
"""
## 📂 Project Structure
├── data/
│   └── NLP_Topic_modeling_Data.csv  # abstracts with 31 discipline labels
├── NMF_TOPIC_MODELING.ipynb         # Full analysis pipeline
└── README.md                        # Documentation
"""
```

## Data Description

The dataset consists of research abstracts across various scientific disciplines. Each entry contains:

- **Abstract**: The main text data used for topic modeling.
- **Fields of Study**: Binary columns indicating the fields of research associated with each abstract (e.g., Physics, Mathematics, Computer Science, etc.).

### Example Columns:

- `id`: Unique identifier for each abstract.
- `ABSTRACT`: The text of the research abstract.
- `Physics`, `Mathematics`, `Statistics`, etc.: Binary indicators of relevance to specific fields.


### Target Variable

The **target variable** for this project is the `ABSTRACT` column, which contains the data used for topic extraction and modeling.

## Methodology

We utilized **Non-negative Matrix Factorization (NMF)** for topic modeling. NMF is a dimensionality reduction technique that factorizes a non-negative matrix ` V ` into two non-negative matrices ` W ` and ` H `, such that:

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\color{white}V\approx&space;W&space;\cdot&space;H" alt="V = W * H">
</p>

- ` V `: Document-term matrix.
- ` W `: Document-topic matrix.
- ` H `: Topic-term matrix.

The optimization problem solved by NMF is:

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\color{white}\min&space;\|V&space;-&space;W&space;H\|_F^2&space;\text{&space;subject&space;to&space;}&space;W\geq&space;0,&space;H\geq&space;0" alt="NMF Optimization">
</p>

Where ` F^2 ` represents the Frobenius norm.

### Topic Coherence

The coherence score measures the semantic similarity between words in a topic. A higher coherence score indicates more interpretable topics. Here’s how it is calculated:

1. **Preprocessing**:
   - The text data is cleaned by removing stop words, punctuation, and irrelevant tokens.
   - The text is tokenized into individual words (or tokens).
   - Words are lemmatized to their root forms.

2. **Topic Extraction**:
   - After applying NMF, each topic is represented as a ranked list of words. These are the most significant words for each topic, determined by their weights in the topic-term matrix ` H `.

3. **Pairwise Word Similarity**:
   - For each topic, pairs of the top ` N ` words are created.
   - A similarity measure, such as Pointwise Mutual Information (PMI), is calculated for each pair based on their co-occurrence in the original dataset.

4. **Average Coherence**:
   - The coherence score for a topic is the average of the pairwise similarities of its words.
   - The overall coherence score ` C ` across all topics is:

   <p align="center">
     <img src="https://latex.codecogs.com/svg.latex?\color{white}C=\frac{1}{N}\sum_{i=1}^{N}\text{Coherence}(\text{Topic}_i)" alt="Coherence formula">
   </p>

   Where ` N ` is the number of topics, and ` Coherence(Topic(i)) ` is the coherence score of the ` i ` topic

### Optimization of Number of Topics

To identify the optimal number of topics ` k `, multiple values were tested. The best ` k ` was chosen based on:
- Maximizing the **coherence score**.
The CoherenceModel from gensim.models evaluates the quality of a topic model by measuring how coherent or semantically meaningful the topics are. It works by comparing the words within each topic and checking their co-occurrence patterns or similarity. It can use different coherence measures, such as C_v, C_p, U_mass, and NPMI, which vary in how they calculate word relationships (e.g., cosine similarity between word vectors or frequency of co-occurrence). The model takes the topic model, the corpus, and the dictionary as inputs and returns a coherence score, where higher values indicate better topic coherence.
- Minimizing the **reconstruction error**.

## Evaluation

The evaluation was conducted using:
1. **Topic Coherence Score**: This ensures that the extracted topics are interpretable and meaningful.
2. **Reconstruction Error**: This measures how well the factorized matrices ` W ` and ` H ` approximate the original matrix ` V `. A lower reconstruction error indicates a better approximation.

## How to Run the Notebook

1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo-name.git
   cd your-repo-name
   ```
2. Installing libraries:
   ```python
     pip install pandas numpy nltk scikit-learn gensim seaborn matplotlib wordcloud
   ```
