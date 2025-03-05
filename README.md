# site-response

## Introduction
This project is about modeling earthquake site response using sequence deep learning models--1D-CNN, LSTM, 
and Transformer. Utilizing ground motion data from the Kiban Kyoshin Network (KiK-net), 
we train these models to predict ground surface acceleration response spectra based on bedrock motions. 
The results are published at https://doi.org/10.3390/app14156658. 
We share the data used in the article in the designsafe data repository. 
We explain details about data in `README.md` included in this repository.


## Install
For linux or wsl2 system,
```shell
# Clone repository
git clone https://github.com/yjchoi1-personal/site-response.git

# Change current directory to the following directory
cd site-response

# Initiate virtual environment
python3 -m virtualenv venv
source venv/bin/activate  # For windows, just `.\venv\Scripts\activate`

# Install requirements
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For windows system,

```shell
# Clone repository
git clone https://github.com/yjchoi1-personal/site-response.git

# Change current directory to the repository
cd site-response

# Initiate virtual environment
python -m venv venv
cmd
cd path\to\site-response
.\venv\Scripts\activate.bat

# Install requirements
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Validate the installation
```shell
python -m pytest test/ -vs
```

## Dataset
> Full dataset is not published yet, but available upon request.

We provide a sample dataset in the `./data` folder. Inside the `./data` folder, you can find `datasets` directory which contains the `FKSH19` site data. It includes the training, validation, and test data, named as `spectrum_train.npz`, `spectrum_valid.npz`, and `spectrum_test.npz`, respectively. The name `FKSH19` corresponds the KiK-net station name (or site name).

## Run
After having the training data at `./data/datasets/`, you can run the model training. The following command trains the `cnn` model. You can change the model type to `lstm` or `transformer` to train other models.

```shell
python main.py \
--config_file config.json \
--site FKSH19 \
--model_id my_cnn_model \
--model_type cnn \
--mode train
```

The model will be saved at `./data/checkpoints/`.

To test the model, run the following command.

```shell
python main.py \
--config_file config.json \
--site FKSH19 \
--model_id my_cnn_model \
--model_type cnn \
--mode test
```

The results will be saved at `./data/outputs/`.

## Arguments
`--config_file` (default: `config.json`)

* Description: Path to the configuration JSON file
* Usage: `--config_file path/to/your/config.json`
* Example: `--config_file configs.json`


`--site` (default: `MYGH04`)

* Description: Specifies the site name for the analysis
* Usage: `--site SITE_NAME`
* Example: `--site FKSH17`


`--model_id` (default: `lstm`)

* Description: the user-defined model name to be saved
* Usage: `--model_id USER_DEFINED_MODEL_NAME`
* Example: `--model_id my_cnn_model`


`--model_type` (default: `lstm`)

* Description: Type of machine learning model to use
* Available options:
  * `cnn`
  * `lstm`
  * `transformer`
* Usage: `--model_type MODEL_TYPE`
* Example: `--model_type cnn`


`--mode` (default: `train`)

* Description: Specifies whether to train a new model or test an existing one
* Available options:
  * train
  * test
* Usage: `--mode MODE`
* Example: `--mode test`
