#!/usr/bin/env sh
set -e

CAFFE=./caffe

SOLVER=./train/adam_solver_5p.prototxt

#WEIGHTS=./models/resnet/train.caffemodel
#SNAPSHOT=./models/resnet/train.solverstate
#
#$CAFFE train --solver=$SOLVER --snapshot=$SNAPSHOT --gpu 0,1,2,3 $@
#$CAFFE train --solver=$SOLVER --weights=$WEIGHTS --gpu 0,1,2,3 $@
$CAFFE train --solver=$SOLVER --gpu 0,1,2,3 $@
