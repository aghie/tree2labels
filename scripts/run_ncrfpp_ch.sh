#Update this path to your virtual environment
source /env/tree2labels/bin/activate

TEST_NAME="test"
TEST_PATH=../CTB_pred_tags/$TEST_NAME"_ch.trees"
USE_GPU=False
INPUT=../dataset/ctb/ctb-$TEST_NAME.seq_lu
EVALB=/home/david.vilares/eclipse-workspace/seq2constree/EVALB/evalb
OUTPUT=../output/
MODELS=../models/
NCRFPP=../NCRFpp/
LOGS=../logs/


#ENRICHED

taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ctb/enriched/bilstm/ctb-bilstm2-chlstm-zzgiga \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/ctb-bilstm2-chlstm-zzgiga.enriched.$TEST_NAME.txt \
--evalb $EVALB \
--ncrfpp $NCRFPP > $LOGS/ctb.enriched.bilstm2.chlstm.zzgiga.cores=1.nogpu=$USE_GPU.$TEST_NAME.log 2>&1


taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ctb/enriched/bilstm/ctb-bilstm2-zzgiga \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/ctb-bilstm2-zzgiga.enriched.$TEST_NAME.txt \
--evalb $EVALB \
--ncrfpp $NCRFPP > $LOGS/ctb.enriched.bilstm2.zzgiga.cores=1.nogpu=$USE_GPU.$TEST_NAME.log 2>&1


#RETAGGER

taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ctb/retagger/bilstm/ctb-bilstm2-chlstm-zzgiga-rt \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/ctb-bilstm2-chlstm-zzgiga.retagger.$TEST_NAME.txt \
--retagger \
--evalb $EVALB \
--ncrfpp $NCRFPP > $LOGS/ctb.retagger.bilstm2.chlstm.zzgiga.cores=1.nogpu=$USE_GPU.$TEST_NAME.log 2>&1

taskset --cpu-list 1 \
python ../run_ncrfpp.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ctb/retagger/bilstm/ctb-bilstm2-zzgiga-rt \
--status test  \
--gpu $USE_GPU \
--output $OUTPUT/ctb-bilstm2-zzgiga.retagger.$TEST_NAME.txt \
--retagger \
--evalb $EVALB \
--ncrfpp $NCRFPP > $LOGS/ctb.retagger.bilstm2.zzgiga.cores=1.nogpu=$USE_GPU.$TEST_NAME.log 2>&1
