#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="3"
python multitask_finetune.py --config=./config/multitask_finetune.json


