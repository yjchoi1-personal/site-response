#!/bin/bash

#data_names=("01_FKSH17" "03_IWTH23" "05_CHBH13" "07_IBRH11" "09_IWTH21" "11_FKSH14" "02_FKSH18" "04_TCGH16" "06_IBRH20" "08_IBRH13" "10_CHBH10" "12_IWTH20")
#data_names=("FKSH17" "IWTH21" "FKSH18" "FKSH19" "IBRH13" "IWTH02" "IWTH05" "IWTH12" "IWTH14" "IWTH22" "IWTH27" "MYGH04")
data_names=("MYGH04")

for item in "${data_names[@]}"; do
  echo ${item}
  python3 tools/make_spectrum_npz.py --dataset_name "${item}"
done