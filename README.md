# Network Intrusion Detection using Machine Learning

This project was developed for the **Machine Learning Models** course and focuses on detecting network intrusions using supervised and unsupervised machine learning techniques on the **NSL-KDD** dataset.

## Project Overview

The main goal of this project is to classify network connections as either:

- **Normal**
- **Attack**

The project includes:

- Data preprocessing and encoding
- Training and evaluation of multiple classification models
- Hyperparameter tuning
- Neural network architecture comparison
- KMeans clustering and comparison with true labels
- Visualization of clustering results using PCA

## Dataset

The dataset used in this project is the **NSL-KDD dataset** (`KDDTrain+.txt`), which is a benchmark dataset for network intrusion detection.
The link to dataset: https://www.kaggle.com/datasets/hassan06/nslkdd


### Features
The dataset contains:
- Numerical features
- Categorical features such as:
  - `protocol_type`
  - `service`
  - `flag`

### Target
The original labels were converted into binary classes:
- `0` = normal
- `1` = attack

## Models Used

The following classification models were implemented and evaluated:

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- Neural Network A
- Neural Network B

## Hyperparameter Tuning

Hyperparameter tuning was performed for:

- Logistic Regression
- Decision Tree
- KNN

For neural networks, two different architectures were tested manually and compared.

## Clustering

The project also includes unsupervised learning using:

- **KMeans clustering**

Different values of `k` were tested, and the clustering results were compared with the real class labels using:

- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)

PCA was used to visualize the clustering results in 2D.

## Project Structure

```text
NETWORK-INTRUSION-DETECTION/
│
├── archive/                 # Dataset files
├── dataset/                 # Additional dataset 
├── notebooks/               # Extra notebooks used during development
├── report/                  # Final report
├── results/                 # Saved figures and CSV results
├── src/                     # Source code files
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   
│
├── main.ipynb               # Main notebook containing the final pipeline
├── requirements.txt         # Required Python libraries
├── README.md                # Project documentation
└── .gitignore
```

## Installation

1. Clone the repository:

```bash
git clone <https://github.com/DelvineBekteshi/Network-Intrusion-Detection.git>
cd NETWORK-INTRUSION-DETECTION
```

2. Install the required libraries:

```bash
pip install -r requirements.txt
```

## Running the Project

Open Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
main.ipynb
```

Run all cells in order:

- Data preprocessing
- Model training and hyperparameter tuning
- Model evaluation
- Clustering
- Saving results

## Output Files

The `results/` folder contains:

- Model performance tables
- Best hyperparameter tables
- Clustering result tables
- PCA clustering plots
- Confusion matrices (if generated)

## Report

The final written report is located in the `report/` folder.

## Requirements

The project uses the following main libraries:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- tensorflow
- jupyter

## Notes

- Make sure the dataset path is correct before running the notebook.
- The main file for execution is `main.ipynb`.
- All experiments and final results are organized in the notebook and saved in the `results/` folder.

## Authors

Flandra Bytyqi
Delvinë Bekteshi