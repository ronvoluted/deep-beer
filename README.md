# Assignment 2: Beer Review Project | Ron Au
Repository for the Beer Review Project from UTS Advanced Data Science for Innovation.

## Docker
### Build image:

```bash
docker build -t deep-beer -f  deployment/Dockerfile .
```

### Run image container

#### Windows (cmd):
```cmd
docker run -itp 8888:8888 -v %cd%:/home/kaggle --name all-star kaggle-nba
```

#### Windows (Powershell):
```cmd
docker run -itp 8888:8888 -v ${PWD}:/home/kaggle --name all-star kaggle-nba
```

#### Mac/Linux:
```bash
docker run -itp 8888:8888 -v "$PWD":/home/kaggle --name all-star kaggle-nba
```

## Project Organization

    ├── Dockerfile         <- Document containing build instructions for Docker image
    ├── LICENSE            <- MIT License
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── Pipfile            <- The requirements file for managing dependency installations
    ├── Pipfile.lock       <- Locks package versions for dependency installations
    ├── README.md          <- The top-level README for developers using this project
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is:
    │                         <lastname>_<firstname>-week<week_number>_<description> e.g.
    │                         wang_kai-ping-week1_1.0-train-data-exploration.ipynb
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

---

Project based on the [cookiecutter data science project template](https://drivendata.github.io/cookiecutter-data-science)
#cookiecutterdatascience
