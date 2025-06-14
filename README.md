# 📧 Spam Email Detection using Machine Learning (TF-IDF + SVM)

This project is a machine learning-based spam detection system that classifies email or SMS messages as either **Spam** or **Not Spam**. It uses a **TF-IDF vectorizer** for feature extraction and a **Support Vector Machine (SVM)** classifier for model training.

---

## 📁 Dataset

We use the **SMS Spam Collection** dataset from [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).  
Upload the dataset (`spam.csv`) to your Google Drive under:


Dataset Columns:
- `v1` → Label (ham or spam)
- `v2` → Message text

---

## 🧠 Technologies Used

| Tool              | Purpose                         |
|-------------------|----------------------------------|
| Python            | Programming Language             |
| Google Colab      | Notebook Environment             |
| Pandas            | Data loading and processing      |
| NLTK              | Text cleaning & stopword removal |
| TF-IDF Vectorizer | Convert text to numerical format |
| SVM (LinearSVC)   | Classification Algorithm         |
| Scikit-learn      | ML tools and metrics             |

---

## 🪜 Project Steps

1. **Mount Google Drive** to access dataset and save model
2. **Load and clean** the dataset (`spam.csv`)
3. **Preprocess text** using regex and stopword removal
4. **Convert messages to TF-IDF vectors**
5. **Train SVM** model on the TF-IDF features
6. **Evaluate** accuracy, precision, recall
7. **Predict custom messages**
8. **Save model & vectorizer** back to Drive

---

## 🧪 Sample Prediction

```python
predict_spam("Congratulations! You have won a free ticket!")
# Output: 🚫 SPAM
```

## 📊 Results
Accuracy: ~98%

![Screenshot 2025-06-14 183424](https://github.com/user-attachments/assets/a8241ad6-1d12-4b9b-b780-16f83e8b8a2c)
