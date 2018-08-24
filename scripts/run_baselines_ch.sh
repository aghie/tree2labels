#Update this path to your virtual environment
source /env/tree2labels/bin/activate

TEST_NAME="test"
INPUT=../dataset/ctb/ctb-$TEST_NAME.seq_lu
TEST_PATH=../CTB_pred_tags/$TEST_NAME"_ch.trees"
USE_GPU=True
EVALB=../EVALB/evalb
OUTPUT=../output/
MODELS=../models/
LOGS=../logs/

#ENRICHED

taskset --cpu-list 1 \
python ../baselines.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ctb/enriched/mlp/ctb-2.2 \
--baseline emlp \
--status test \
--gpu $USE_GPU \
--output_decode $OUTPUT/ctb-2.2.emlp.enriched.$TEST_NAME.txt \
--evalb $EVALB > $LOGS/ctb.enriched.emlp.cores=1.nogpu=$USE_GPU.$TEST_NAME.log 2>&1

#RETAGGER

taskset --cpu-list 1 \
python ../baselines.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ctb/retagger/mlp/ctb-2.2-rt \
--baseline emlp \
--status test \
--gpu $USE_GPU \
--output_decode $OUTPUT/ctb-2.2.emlp.retagger.$TEST_NAME.txt \
--retagger \
--evalb $EVALB > $LOGS/ctb.retagger.emlp.cores=1.nogpu=$USE_GPU.$TEST_NAME.log 2>&1

