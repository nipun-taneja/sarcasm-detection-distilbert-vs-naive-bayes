# Sarcasm Detection with DistilBERT vs Naive Bayes

This project evaluates sarcasm detection on news headlines using two approaches:
1. A **Naive Bayes baseline** with bag-of-words features.
2. A **fine-tuned DistilBERT model** for binary classification.

The notebook integrates both methods, compares their performance, and analyzes trade-offs.

---

## Objective
- Train and evaluate a Naive Bayes classifier on the sarcasm dataset.
- Fine-tune DistilBERT (`distilbert-base-uncased`) for the same task.
- Compare metrics (Accuracy, Precision, Recall, F1) across models.
- Visualize results and analyze errors.
- Discuss trade-offs between classical and transformer-based NLP.

---

## Data
- **Dataset:** [News Headlines Dataset for Sarcasm Detection (Kaggle)](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection)
- Each row:  
  - `headline`: news headline text  
  - `is_sarcastic`: 1 = sarcastic, 0 = not sarcastic
- Kaggle login required for access.

---

## Methodology
- **Naive Bayes:** Bag-of-words representation, simple probabilistic baseline.
- **DistilBERT:** Transformer fine-tuning with a classification head.
- **Evaluation metrics:** Accuracy, Precision, Recall, F1, confusion matrices, ROC/PR curves.
- **Error analysis:** Review high-confidence false positives/negatives.

---

## Results

### Performance Summary
| Model                 | Accuracy | Precision | Recall | F1  |
|------------------------|---------:|----------:|-------:|----:|
| Naive Bayes            |    0.79  |     0.79  |   0.78 | 0.79 |
| DistilBERT (full set)  |    0.93  |     0.93  |   0.92 | 0.93 |
| DistilBERT (small set) |    0.86  |     0.86  |   0.86 | 0.86 |

### Confusion Matrices
<img width=45% height="547" alt="image" src="https://github.com/user-attachments/assets/a35ea557-e303-4d2b-a96d-9d9fe33f756b" />

![DistilBERT Confusion Matrix](images/confusion_matrix_distilbert.png)


<img width=45% height="590" alt="image" src="https://github.com/user-attachments/assets/399178b1-afc0-4ca7-aad9-430ee82d9831" />
<img width=45% height="590" alt="image" src="https://github.com/user-attachments/assets/f58a1b81-bdb9-41a6-b98b-8cf37830f85f" />

<img width=45% height="590" alt="image" src="https://github.com/user-attachments/assets/2152f685-908b-4dfc-b945-1bf89f8ed00d" />
<img width=45% height="590" alt="image" src="https://github.com/user-attachments/assets/79ce0998-856f-472b-8aa6-b920d16f0b3f" />

<img width="856" height="514" alt="image" src="https://github.com/user-attachments/assets/87988cb6-8774-4fab-b28c-dd587e7b848f" />




---

## Key Findings
- DistilBERT outperforms Naive Bayes across all metrics, especially recall and F1.
- Even with reduced training data, DistilBERT surpasses Naive Bayes.
- Naive Bayes is faster and simpler but misses context-dependent sarcasm.
- DistilBERT is more robust but requires more computational resources.

---

## Practical Trade-offs
- **Naive Bayes:** Lightweight, interpretable, suitable as a baseline or in constrained environments.
- **DistilBERT:** More accurate and robust, better suited for production sarcasm detection where nuanced understanding is required.

---

## How to Run
Open in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nipun-taneja/sarcasm-detection-distilbert-vs-naive-bayes/blob/main/notebooks/Sarcasm_Detection_Project_Cleaned.ipynb)

Steps:
1. Ensure Kaggle dataset access (see notebook instructions).
2. Run all cells in Colab.
3. Artifacts (reports, images, predictions) are generated automatically.

---

## Repository Structure
