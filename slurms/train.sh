#!/bin/bash

sites=("MYGH04")
#sites=("FKSH17" "IWTH21" "FKSH18" "FKSH19" "IBRH13" "IWTH02" "IWTH05" "IWTH12" "IWTH14" "IWTH22" "IWTH27" "MYGH04")
models=("lstm" "cnn" "transformer")

for site in "${sites[@]}";
do
  for model in "${models[@]}"
  do
    echo ${site}
    echo ${model}
    python3 main.py \
      --mode train \
      --config_file config.json \
      --site ${site} \
      --model_id "${model}" \
      --model_type ${model}
  done
done


