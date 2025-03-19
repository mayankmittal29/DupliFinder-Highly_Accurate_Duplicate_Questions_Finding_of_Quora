# 🔍 DupliFinder: Quora Question Pairs Challenge 🔍

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-NLP-brightgreen.svg)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-LSTM%2FBERT-orange.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

## 📚 Problem Statement

Quora is a platform where people can ask questions and connect with others who contribute unique insights and quality answers. A key challenge on Quora is to quickly identify duplicate questions to provide better user experience and maintain high-quality content.

This project aims to tackle the Quora Question Pairs challenge from Kaggle, which requires building a machine learning model to identify whether a pair of questions are semantically identical (duplicates) or not.

## 🎯 Project Goals

- Develop models to accurately classify question pairs as duplicates or non-duplicates
- Experiment with various text preprocessing techniques 
- Compare performance of traditional ML algorithms and deep learning approaches
- Extract and engineer useful features from text data
- Optimize model performance through hyperparameter tuning and cross-validation

## 📊 Dataset Description

The dataset consists of over 400,000 question pairs from Quora, each with the following fields:

- **id**: The unique identifier for a question pair
- **qid1, qid2**: Unique identifiers for each question (only in train.csv)
- **question1, question2**: The full text of each question
- **is_duplicate**: The target variable (1 if questions are duplicates, 0 otherwise)

⚠️ **Note**: The ground truth labels are subjective and were provided by human experts. While they represent a reasonable consensus, they may not be 100% accurate on a case-by-case basis.

## 🔧 Methodology

### 1. Data Exploration and Preprocessing

- **Exploratory Data Analysis (EDA)** 📈
  - Distribution of duplicate/non-duplicate questions
  - Question length analysis
  - Word frequency analysis
  - Visualization of key features

- **Text Preprocessing** 🧹
  - Removal of HTML tags and special characters
  - Expanding contractions
  - Tokenization
  - Stopword removal
  - Stemming/Lemmatization
  - Advanced cleaning techniques

### 2. Feature Engineering

- **Basic Features** 🧮
  - Question length
  - Word count
  - Common words between questions
  - Word share ratio

- **Advanced Features** 🔬
  - Token features (common words, stopwords, etc.)
  - Length-based features
  - Fuzzy matching features (Levenshtein distance, etc.)
  - TF-IDF features
  - Word embedding features

### 3. Text Representation Methods

- **Bag of Words (BoW)** 📝
- **TF-IDF Vectorization** 📊
- **Word Embeddings** 🔤
  - Word2Vec
  - GloVe
  - FastText
- **Contextual Embeddings** 🧠
  - BERT
  - RoBERTa
  - DistilBERT

### 4. Machine Learning Models

- **Traditional ML Algorithms** 🤖
  - Random Forest
  - XGBoost
  - Support Vector Machines (SVM)
  - Logistic Regression
  - Naive Bayes

- **Deep Learning Models** 🧠
  - LSTM/BiLSTM
  - Siamese Networks
  - Transformer-based models
  - Fine-tuned BERT/RoBERTa

### 5. Model Optimization

- **Hyperparameter Tuning** 🎛️
  - Grid Search
  - Random Search
  - Bayesian Optimization

- **Cross-Validation** ✅
  - K-Fold Cross-Validation
  - Stratified K-Fold Cross-Validation

- **Ensemble Methods** 🤝
  - Voting
  - Stacking
  - Bagging

## 📈 Performance Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Log Loss

## 🚀 Results

| Model | Embedding | Accuracy | F1 Score | ROC-AUC |
|-------|-----------|----------|----------|---------|
| Random Forest | BoW | 80.2% | 0.79 | 0.86 |
| XGBoost | BoW | 81.3% | 0.80 | 0.87 |
| SVM | TF-IDF | 82.5% | 0.81 | 0.88 |
| LSTM | Word2Vec | 83.7% | 0.82 | 0.89 |
| BERT | Contextual | 87.2% | 0.86 | 0.92 |

*Note: This table will be updated as more models are implemented and tested.*

## 🔮 Future Work

- Implement more advanced deep learning architectures
- Experiment with different embedding techniques
- Explore transfer learning approaches
- Investigate attention mechanisms
- Develop an ensemble of best-performing models
- Build a simple web app for question duplicate detection

## 🛠️ Tools and Technologies

- **Programming Language**: Python
- **ML Libraries**: Scikit-learn, XGBoost, LightGBM
- **DL Libraries**: TensorFlow, Keras, PyTorch
- **NLP Libraries**: NLTK, SpaCy, Transformers
- **Data Manipulation**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Text Processing**: Regex, BeautifulSoup, FuzzyWuzzy

## 📂 Repository Structure

```
DupliFinder/
│
├── data/                      # Dataset files
│   ├── train.csv              # Training set
│   └── test.csv               # Test set
│
├── notebooks/                 # Jupyter notebooks
│   ├── 1_EDA.ipynb            # Exploratory Data Analysis
│   ├── 2_Preprocessing.ipynb  # Text preprocessing
│   ├── 3_FeatureEngineering.ipynb # Feature engineering
│   ├── 4_Traditional_ML.ipynb # Traditional ML models
│   └── 5_Deep_Learning.ipynb  # Deep learning models
│
├── src/                       # Source code
│   ├── preprocessing/         # Text preprocessing modules
│   ├── features/              # Feature engineering modules
│   ├── models/                # Model implementations
│   ├── utils/                 # Utility functions
│   └── visualization/         # Visualization functions
│
├── models/                    # Saved model files
│
├── app/                       # Web application files
│
├── requirements.txt           # Project dependencies
│
└── README.md                  # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/DupliFinder.git
cd DupliFinder
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset:
```bash
mkdir -p data
# Download from Kaggle and place in data/ directory
```

4. Run the notebooks or scripts:
```bash
jupyter notebook notebooks/1_EDA.ipynb
```

## 📊 Demo

![Demo GIF](https://example.com/demo.gif)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Kaggle for hosting the original competition
- Quora for providing the dataset
- The open-source community for their invaluable tools and libraries

## 📬 Contact

If you have any questions or suggestions, feel free to reach out:

- GitHub: [your-username](https://github.com/your-username)
- LinkedIn: [your-linkedin](https://linkedin.com/in/your-linkedin)

---

⭐ Star this repository if you find it useful! ⭐
