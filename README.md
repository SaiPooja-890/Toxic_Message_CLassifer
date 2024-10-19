# Toxic_Message_CLassifer


This repository contains a Jupyter notebook, **`Toxic_Classifier.ipynb`**, which implements a machine learning model to classify toxic comments or text. The project leverages NLP techniques to detect various forms of abusive, offensive, or inappropriate language.

## 📑 Overview  

Toxic comments can harm online communities and platforms. This notebook demonstrates how to build a **text classifier** that can identify toxic content. It covers essential steps, from **data preprocessing and feature extraction** to **training** and **evaluating** the model.

### Key Features:
- **Text Preprocessing**: Tokenization, stopword removal, and text normalization.
- **Tokenizer Usage**: Uses a pretrained tokenizer (like `tokenizer.pkl`).
- **Model Training**: Trains a classifier (e.g., Logistic Regression, SVM, or Neural Network) to detect toxic text.
- **Evaluation Metrics**: Includes accuracy, precision, recall, and F1-score.

## 🛠️ Prerequisites  
Before running the notebook, ensure the following dependencies are installed:  
```bash
pip install numpy pandas scikit-learn nltk
```

## 📂 Files in this Repository  
- **`Toxic_Classifier.ipynb`**: Main notebook containing the toxic comment classification code.  
- **`tokenizer.pkl`**: Pretrained tokenizer file used for feature extraction.
- **`logistic_regression_model.pkl`**: Trained Model using Logistic Regression.
- **`random_forest_model.pkl`**:  Trained Model using Random-Forest.
- **`toxicity_en.csv`**: Dataset for training the model.


## 🚀 How to Use  
1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. **Open the Jupyter Notebook:**
   ```bash
   jupyter notebook Toxic_Classifier.ipynb
   ```
3. **Run the cells sequentially** to preprocess data, train the model, and evaluate the results.

## 📊 Example Output  
The model is trained to detect the following classes:
- **Toxic**
- **Severe Toxic**
- **Obscene**
- **Threat**
- **Insult**
- **Identity Hate**

Sample predictions:
| Comment                         | Prediction   |
|---------------------------------|--------------|
| "You are so stupid!"            | Toxic        |
| "This is a great post!"         | Non-Toxic    |

## Screenshots of our model performance and output :

![WhatsApp Image 2024-10-19 at 3 20 27 PM](https://github.com/user-attachments/assets/f6595e39-a44b-4814-9be7-94c90e40b226)

![WhatsApp Image 2024-10-19 at 3 20 38 PM](https://github.com/user-attachments/assets/908f164a-6085-407a-939f-073ad52d42b3)

![WhatsApp Image 2024-10-19 at 3 20 56 PM](https://github.com/user-attachments/assets/087d9f98-a058-4fd4-ab77-7b1a07268af1)


