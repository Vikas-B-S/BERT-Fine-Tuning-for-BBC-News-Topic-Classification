# News Headline Classifier (Fine-Tuned)

This project fine-tunes a Transformer-based model to classify BBC news
headlines into five categories: **business**, **entertainment**,
**politics**, **sport**, and **tech**.

------------------------------------------------------------------------

## 📁 Project Structure

    News_Headline_Classifier_fine_tuned/
    │── src/                # Training, inference, data loading scripts
    │── config/             # Config files (paths, hyperparameters)
    │── results/            # Saved model
    │── venv/               # Python virtual environment (fine_tuned)
    │── requirements.txt    # Project dependencies
    │── README.md           # Project documentation

------------------------------------------------------------------------

## 🚀 Features

-   Fine-tuned Transformer model for news headline classification\
-   Cleaned and validated BBC dataset\
-   Class distribution checked for potential imbalance\
-   Evaluation pipeline\
-   Inference script for predicting categories for new headlines

------------------------------------------------------------------------

## 📦 Installation

### 1. Clone the repository

    git clone <your_repo_url>
    cd News_Headline_Classifier_fine_tuned

### 2. Create & activate virtual environment

    python -m venv fine_tuned
    fine_tuned\Scripts\activate

### 3. Install dependencies

    pip install -r requirements.txt

------------------------------------------------------------------------

## 🏋️‍♂️ Training

To train the model:

    python -m src.train

This will: - Load the BBC dataset\
- Encode labels\
- Train the Transformer classifier\
- Save model in `results/`

------------------------------------------------------------------------

## 🔍 Evaluate

To evaluate the model on test data:

    python -m src.evaluate_test

------------------------------------------------------------------------

## 🔍 Inference

Run inference on new headlines:

    python -m src.inference

Example:

    Input: "Stock markets hit record highs"
    Output: business

------------------------------------------------------------------------

## 📊 Dataset Class Distribution

The project includes a class distribution check:

-   politics -- 275\
-   entertainment -- 286\
-   sport -- 210\
-   business -- 212\
-   tech -- 242

No major imbalance issues.

------------------------------------------------------------------------

## 🗂️ Label Mapping

    0 → business
    1 → entertainment
    2 → politics
    3 → sport
    4 → tech

This mapping is consistent across training and inference.

------------------------------------------------------------------------


## 📈 Results

After training, evaluation metrics (accuracy/F1) are saved in
`/results/metrics.json`.

------------------------------------------------------------------------

## 🙌 Author

Created as part of machine learning fine‑tuning practice.

------------------------------------------------------------------------