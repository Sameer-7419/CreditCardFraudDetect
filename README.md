# Credit Card Fraud Detection with Logistic Regression

This project uses a machine learning approach to detect fraudulent credit card transactions using the **Logistic Regression** algorithm. The dataset is highly imbalanced, so **under-sampling** is performed to balance the classes for better model performance.

## 📂 Dataset

The dataset used is `creditcard.csv`, which contains anonymized features of transactions and a `Class` column:
- `0`: Legitimate Transaction
- `1`: Fraudulent Transaction

Source: [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## 📊 Exploratory Data Analysis

- The dataset is highly imbalanced (~284,000 legit vs. 492 fraud transactions).
- Fraudulent transactions generally involve smaller amounts.
- Group-wise feature analysis was done using `.groupby('Class').mean()`.

---

## ⚙️ Data Preprocessing

To handle class imbalance:
- **Under-sampling** was applied:
  - Randomly sampled 492 legitimate transactions
  - Combined with the 492 fraud transactions
  - Resulting in a balanced dataset with 984 records

The features (`X`) and target (`Y`) were extracted, then split into:
- **80% training**
- **20% testing**
(using `train_test_split` with `stratify=Y` to preserve class distribution)

---

## 🧠 Model Training

- **Algorithm:** Logistic Regression (`sklearn.linear_model`)
- Trained on the balanced dataset
- No feature scaling applied (can be added later for performance boost)

---

## 📈 Evaluation

Model performance was evaluated using accuracy:

- ✅ **Training Accuracy:** `94.91%`
- ✅ **Testing Accuracy:** `95.43%`

### Model Fit Insight:
- The training and testing accuracies are very close
- 🔍 **Conclusion:** The model is well-fitted — not overfitting or underfitting

---

## 🛠 Requirements

Install the necessary libraries:

```bash
pip install pandas numpy scikit-learn
