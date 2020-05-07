#!/bin/bash

# Allow default values to be defined if vars not defined
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export DUMP_PATH=${DUMP_PATH:-/data/ozan/experiments/xlm_mmvc}
export DATA_PATH=${DATA_PATH:-/data2/ozan/conceptual_captions/mmvc_icl_data/parallel.tok.bpe}
export FEAT_PATH=/data2/ozan/conceptual_captions/avgpool_features

mkdir -p $DUMP_PATH

PAIR=$(basename `ls ${DATA_PATH}/train.*pth | head -n1` | cut -d'.' -f2)
L1=`echo $PAIR | cut -d'-' -f1`
EPOCH=`wc -l ${DATA_PATH}/train.${PAIR}.$L1 | head -n1 | cut -d' ' -f1`
EMB=${EMB:-512}
NL=${NL:-6}
NH=${NH:-8}
BS=${BS:-512}
LR=${LR:-0.0005}
NAME="concap_emb${EMB}_${NL}l_${NH}h_bs${BS}_lr${LR}"
echo $NAME

ipython -i train.py -- --exp_name $NAME --dump_path $DUMP_PATH --data_path $DATA_PATH \
--emb_dim ${EMB} --n_layers ${NL} --n_heads ${NH} \
--dropout 0.1 --attention_dropout 0.1 --gelu_activation true --batch_size ${BS} --bptt 256 \
--optimizer "adam,lr=${LR}" --epoch_size ${EPOCH} --max_epoch 100000 \
--validation_metrics _valid_en_de_mlm_ppl --stopping_criterion _valid_en_de_mlm_ppl,50 \
--lgs 'en-de' --clm_steps '' --mlm_steps 'en-de' --only_vlm true --region_feats_path ${FEAT_PATH} \
--image_names ${DATA_PATH} --fp16 false --save_periodic 2 $@