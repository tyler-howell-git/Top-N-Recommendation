# Top-N Movie Recommendation System Project (COMP 7118)
### Author: Tyler Howell

This project implements and compares three Top-N movie recommendation approaches using the MovieLens dataset:

1. Popularity-based baseline
2. Item-item kNN collaborative filtering
3. SVD matrix factorization using Surprise

The main demo is contained in the Jupyter notebook. The supporting Python files are organized in the project root for simplicity.

## Project Structure

```text
project-root/
|
|- dataset/
|   |- ratings.csv
|   |- movies.csv
|
|- data_preprocessing.py
|- models.py
|- evaluate.py
|
|- demo.ipynb
|- README.md
```
# Dataset

The MovieLens dataset was used to evaluate the recommenders and can be downloaded [here.](https://grouplens.org/datasets/movielens/)  
Upon arrival to the main MovieLens page:
- Download the dataset the zip file titled "ml-latest-small.zip"
- Extract the dataset from the zip and rename to "dataset"
- Place the "dataset" file in project root as diagrammed above

# Environment Setup

This project uses Python 3.10 because `scikit-surprise` has compatibility issues with newer Python and NumPy versions.

Create a new conda environment:

```bash
conda create -n top-n python=3.10
```

Activate the environment:

```bash
conda activate top-n
```

Install required packages:

```bash
pip install "numpy<2" pandas scikit-learn scikit-surprise matplotlib notebook ipykernel
```

Register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name top-n --display-name "Python (top-n)"
```

# Running the Project

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open the notebook:

```text
demo.ipynb
```

Select the kernel:

```text
Python (top-n)
```

Run the notebook from top to bottom.
