#!/bin/sh

papermill --log-output -k python3 main_train.ipynb runs/main_train_03.ipynb
papermill --log-output -k python3 main_evaluation.ipynb runs/main_evaluation_03.ipynb