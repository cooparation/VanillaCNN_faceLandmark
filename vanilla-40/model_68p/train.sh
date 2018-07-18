#!/usr/bin/env sh
set -e

CAFFE=./caffe
SOLVER=./vanilla-40/model_68p/vanilla_adam_solver.prototxt
WEIGHTS=./vanilla-40/model_68p/_iter_1400000.caffemodel
#SNAPSHOT=_iter_110000.solverstate

#$CAFFE train  --solver=$SOLVER --snapshot=$SNAPSHOT --gpu 1 \
#$CAFFE train --solver=$SOLVER --gpu 1 \
$CAFFE train  --solver=$SOLVER --weights=$WEIGHTS --gpu 1 \
    2>&1 | tee output_tmp/train_68p_log.log
