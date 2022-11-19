#!/bin/bash

#python train.py --img 3072 --batch 4 --epochs 10 --data data/GBR_Starfish_Detection.yaml --weights ./runs/train/GBR-Starfish-Detect-allImg/weights/best.pt --name GBR-Starfish-Detect-allImg-ext --hyp data/hyps/GBR_Startfish_Hpy.yaml

IMG=3072
BATCH=3
EPOCH=15
CFG="/home/xyb/Project/Kaggle/GreatBarrierReef/GBR_data_set_txt"
WEIGHT="./yolov5m6.pt"
HYP="data/hyps/GBR_Startfish_Hpy.yaml"
BASE_NAME="GBR-COTs-GrpKFold"

OPTIMIZER='Adam'

printf ">>>>> SHELL_COMMAND:: Using Group:$1 >>>>> \n"
cnt=$1
#python train.py --img $IMG \
#	--batch $BATCH \
#	--epochs $EPOCH \
#	--data "${CFG}/gbr-${cnt}.yaml" \
#	--weights $WEIGHT \
#	--name "${BASE_NAME}-${cnt}-Adam" \
#	--hyp $HYP \
#	--optimizer $OPTIMIZER

python train.py --img $IMG \
	--batch $BATCH \
	--epochs $EPOCH \
	--data "${CFG}/gbr-${cnt}.yaml" \
	--weights $WEIGHT \
	--name "${BASE_NAME}-${cnt}-SGD-M-weight-allImgInT-NoEmptyInV" \
	--hyp $HYP

##for cnt in {0..4}
##do
##    printf ">>>>> SHELL_COMMAND:: Loop:$cnt >>>>> \n"
##    if [ $cnt -eq 0 ]
##    then
##        python train.py --img $IMG --batch $BATCH --epochs $EPOCH --data "${CFG}/gbr-${cnt}.yaml" --weights "./yolov5s6.pt" --name "${BASE_NAME}-${cnt}" --hyp $HYP
##    else
##        index=`expr $cnt - 1`
##        python train.py --img $IMG --batch $BATCH --epochs $EPOCH --data "${CFG}/gbr-${cnt}.yaml" --weights "./runs/train/${BASE_NAME}-${index}/weights/last.pt" --name "${BASE_NAME}-${cnt}" --hyp $HYP
##    fi
##
##done

