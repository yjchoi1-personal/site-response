# site-response

## Introduction
This project is about modeling earthquake site response using sequence deep learning models--1D-CNN, LSTM, 
and Transformer. Utilizing ground motion data from the Kiban Kyoshin Network (KiK-net), 
we train these models to predict ground surface acceleration response spectra based on bedrock motions. 
The results are published at https://doi.org/10.3390/app14156658. 
We share the data used in the article in the designsafe data repository. 
We explain details about data in `README.md` included in this repository.


## Install
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

## Download dataset
> Dataset is not published yet.

In the data folder, `datasets` and `datasets_publish` contains the same raw data for KiK-net 
and result from SHAKE2000. However, `datasets` are the directory that was actaully used for training code 
which contains some experimental dummy data. `datasets_publish` is clearned version of `datasets` 
which only contains raw data. 

The folder names like `FKSH17`, `IWTH02`, ..., correspond to KiK-net station (or site). 
Each folder contains `data_all_x` and `data_all_y`, 
which contains bedrock measurements and ground surface measurements in `.csv` format, respectively, 
for 100 earthquake events. 
Each pair of `csv` comprises a single training example. 
For example, in `FKSH17`, `data_all_x/FKSH17_1001.csv` is a bedrock measurement for a single event, 
and `data_all_y/FKSH17_2101.csv` is the ground response for the corresponding event. 

For each site, we process these time domain `.csv` files to frequency domain, 
and split those into train, test, and validation set using `.npz` format. 
The deep learning models are trained for each site to consider site-specific response. 


## Run
After having the training data at `./data` you can run the `lstm` model training as follows.
```shell
python train.py \
--config_file config.json \
--site MYGH04 \
--model_id my_lstm \
--model_type lstm \
--mode train
```

## Instruction
`--config_file` (default: `config.json`)

* Description: Path to the configuration JSON file
* Usage: `--config_file path/to/your/config.json`
* Example: `--config_file configs.json`


`--site` (default: `MYGH04`)

* Description: Specifies the site name for the analysis
* Usage: `--site SITE_NAME`
* Example: `--site FKSH17`


`--model_id` (default: `lstm`)

* Description: Unique identifier for saving model results
* Usage: `--model_id MODEL_IDENTIFIER`
* Example: `--model_id lstm_experiment1`


`--model_type` (default: `lstm`)

* Description: Type of machine learning model to use
* Available options:
  * cnn (Convolutional Neural Network)
  * lstm (Long Short-Term Memory)
  * transformer
* Usage: `--model_type MODEL_TYPE`
* Example: `--model_type cnn`


`--mode` (default: `train`)

* Description: Specifies whether to train a new model or test an existing one
* Available options:
  * train
  * test
* Usage: `--mode MODE`
* Example: `--mode test`
