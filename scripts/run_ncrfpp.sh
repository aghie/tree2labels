#Update this path to your virtual environment
source /env/tree2labels/bin/activate


TEST_NAME="test"
INPUT=../dataset/ptb/ptb-$TEST_NAME.seq_lu
TEST_PATH=../PTB_pred_tags/$TEST_NAME.trees
USE_GPU=False
EVALB=../EVALB/evalb
OUTPUT=../output/
MODELS=../models/
NCRFPP=../NCRFpp/
LOGS=../logs/

#ENRICHED

taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ptb/enriched/bilstm/ptb-bilstm2-chlstm-glove \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/ptb-bilstm2-chlstm-glove.enriched.$TEST_NAME.txt \
--evalb $EVALB \
--ncrfpp $NCRFPP > $LOGS/ptb.enriched.bilstm2.chlstm.glove.cores=1.nogpu=$USE_GPU.$TEST_NAME.log 2>&1


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ptb/enriched/bilstm/ptb-bilstm2-glove \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/ptb-bilstm2-glove.enriched.$TEST_NAME.txt \
--evalb $EVALB \
--ncrfpp $NCRFPP > $LOGS/ptb.enriched.bilstm2.glove.cores=1.nogpu=$USE_GPU.$TEST_NAME.log 2>&1


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ptb/enriched/bilstm/ptb-bilstm1-chlstm-glove \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/ptb-bilstm1-chlstm-glove.enriched.$TEST_NAME.txt \
--evalb $EVALB \
--ncrfpp $NCRFPP > $LOGS/ptb.enriched.bilstm1.chlstm.glove.cores=1.nogpu=$USE_GPU.$TEST_NAME.log 2>&1


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ptb/enriched/bilstm/ptb-bilstm1-glove \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/ptb-bilstm1-glove.enriched.$TEST_NAME.txt \
--evalb $EVALB \
--ncrfpp $NCRFPP > $LOGS/ptb.enriched.bilstm1.glove.cores=1.nogpu=$USE_GPU.$TEST_NAME.log 2>&1


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ptb/enriched/bilstm/ptb-bilstm1 \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/ptb-bilstm1.enriched.$TEST_NAME.txt \
--evalb $EVALB \
--ncrfpp $NCRFPP > $LOGS/ptb.enriched.bilstm1.cores=1.nogpu=$USE_GPU.$TEST_NAME.log 2>&1


#RETAGGER

taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ptb/retagger/bilstm/ptb-bilstm2-chlstm-glove-rt \
--status test \
--gpu $USE_GPU \
--retagger \
--output $OUTPUT/ptb-bilstm2-chlstm-glove.retagged.$TEST_NAME.txt \
--evalb $EVALB \
--ncrfpp $NCRFPP > $LOGS/ptb.retagger.bilstm2.chlstm.glove.cores=1.nogpu=$USE_GPU.$TEST_NAME.log 2>&1


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ptb/retagger/bilstm/ptb-bilstm2-glove-rt \
--status test \
--gpu $USE_GPU \
--retagger \
--output $OUTPUT/ptb-bilstm2-glove.retagged.$TEST_NAME.txt \
--evalb $EVALB \
--ncrfpp $NCRFPP > $LOGS/ptb.retagger.bilstm2.glove.cores=1.nogpu=$USE_GPU.$TEST_NAME.log 2>&1


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ptb/retagger/bilstm/ptb-bilstm1-chlstm-glove-rt \
--status test \
--gpu $USE_GPU \
--retagger \
--output $OUTPUT/ptb-bilstm1-chlstm-glove.retagged.$TEST_NAME.txt \
--evalb $EVALB \
--ncrfpp $NCRFPP > $LOGS/ptb.retagger.bilstm1.chlstm.glove.cores=1.nogpu=$USE_GPU.$TEST_NAME.log 2>&1


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ptb/retagger/bilstm/ptb-bilstm1-glove-rt \
--status test \
--gpu $USE_GPU \
--retagger \
--output $OUTPUT/ptb-bilstm1-glove.retagged.$TEST_NAME.txt \
--evalb $EVALB \
--ncrfpp $NCRFPP > $LOGS/ptb.retagger.bilstm1.glove.cores=1.nogpu=$USE_GPU.$TEST_NAME.log 2>&1


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ptb/retagger/bilstm/ptb-bilstm1-rt \
--status test \
--gpu $USE_GPU \
--retagger \
--output $OUTPUT/ptb-bilstm1.retagged.$TEST_NAME.txt \
--evalb $EVALB  \
--ncrfpp $NCRFPP > $LOGS/ptb.retagger.bilstm1.cores=1.nogpu=$USE_GPU.$TEST_NAME.log 2>&1
