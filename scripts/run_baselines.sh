#Update this path to your virtual environment
source /env/tree2labels/bin/activate

TEST_NAME="test"
INPUT=../dataset/ptb/ptb-$TEST_NAME.seq_lu
TEST_PATH=../PTB_pred_tags/$TEST_NAME.trees
USE_GPU=False
EVALB=../EVALB/evalb
OUTPUT=../output/
MODELS=../models/
LOGS=../logs/

###################################################################################
#			             PTB          				  #
###################################################################################


#ENRICHED


taskset --cpu-list 1 \
python ../baselines.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ptb/enriched/crf/ptb-1.1 \
--baseline crf \
--status test \
--gpu $USE_GPU \
--output_decode $OUTPUT/ptb-1.1.crf.enriched.$TEST_NAME.txt \
--evalb $EVALB > $LOGS/ptb.enriched.crf.cores=1.nogpu=$USE_GPU.$TEST_NAME.log 2>&1


taskset --cpu-list 1 \
python ../baselines.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ptb/enriched/mlp/ptb-2.2 \
--baseline emlp \
--status test \
--gpu $USE_GPU \
--output_decode $OUTPUT/ptb-2.2.emlp.enriched.$TEST_NAME.txt \
--evalb $EVALB > $LOGS/ptb.enriched.emlp.cores=1.nogpu=$USE_GPU.$TEST_NAME.log 2>&1


taskset --cpu-list 1 \
python ../baselines.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ptb/enriched/mlp/ptb-1.1 \
--baseline mlp \
--status test \
--gpu $USE_GPU \
--output_decode $OUTPUT/ptb-1.1.mlp.enriched.$TEST_NAME.txt \
--evalb $EVALB > $LOGS/ptb.enriched.mlp.cores=1.nogpu=$USE_GPU.$TEST_NAME.log 2>&1


#RETAGGER

taskset --cpu-list 1 \
python ../baselines.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ptb/retagger/mlp/ptb-2.2-rt \
--baseline emlp \
--status test \
--retagger \
--gpu $USE_GPU \
--output_decode $OUTPUT/ptb-2.2.emlp.retagger.$TEST_NAME.txt \
--evalb $EVALB > $LOGS/ptb.retagger.emlp.cores=1.nogpu=$USE_GPU.$TEST_NAME.log 2>&1


taskset --cpu-list 1 \
python ../baselines.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ptb/retagger/mlp/ptb-1.1-rt \
--baseline mlp \
--status test \
--retagger \
--gpu $USE_GPU \
--output_decode $OUTPUT/ptb-1.1-rt.mlp.retagger.$TEST_NAME.txt \
--evalb $EVALB > $LOGS/ptb.retagger.mlp.cores=1.nogpu=$USE_GPU.$TEST_NAME.log 2>&1


taskset --cpu-list 1 \
python ../baselines.py \
--test $INPUT \
--gold $TEST_PATH \
--model $MODELS/ptb/retagger/crf/ptb-1.1-rt \
--baseline crf \
--status test \
--retagger \
--gpu $USE_GPU \
--output_decode $OUTPUT/ptb-1.1.crf.retagger.$TEST_NAME.txt \
--evalb $EVALB > $LOGS/ptb.retagger.crf.cores=1.nogpu=$USE_GPU.$TEST_NAME.log 2>&1




