#!/usr/bin/env sh
set -e

CAFFE=./caffe
SOLVER=./training/net2/vanilla_adam_solver_1.prototxt
WEIGHTS=./model/net2/_iter_150000.caffemodel
#SNAPSHOT=_iter_110000.solverstate

#$CAFFE train  --solver=$SOLVER --snapshot=$SNAPSHOT --gpu 1 \
#$CAFFE train  --solver=$SOLVER --weights=$WEIGHTS --gpu 1 \
$CAFFE train --solver=$SOLVER --gpu 1 \
    2>&1 | tee output_tmp/train_5p_log.log
