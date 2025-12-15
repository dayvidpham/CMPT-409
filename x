#!/usr/bin/env bash

python3 run.py --output prayers/soudry_gd --preset soudry_gd
python3 run.py --output prayers/soudry_sgd --preset soudry_sgd
python3 run.py --output prayers/adam_gd --preset adam_gd
python3 run.py --output prayers/adam_sgd --preset adam_sgd
python3 run.py --output prayers/twolayer_gd \
    --model twolayer \
    --optimizer-family gd \
    --deterministic True \
    --loss logistic
python3 run.py --output prayers/twolayer_sgd \
    --model twolayer \
    --optimizer-family gd \
    --deterministic False \
    --loss logistic
