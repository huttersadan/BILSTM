MODEL_NAME="NCBI-oursnew1"
RAW_TEXT="data/NCBI-oursnew1/raw_text.txt"
DICT_CORE="data/NCBI-oursnew1/dict_core.txt"
DICT_FULL="data/NCBI-oursnew1/dict_full.txt"
EMBEDDING_TXT_FILE="embedding/bio_embedding.txt"
MUST_RE_RUN=0

green=`tput setaf 2`
reset=`tput sgr0`

DEV_SET="data/NCBI-oursnew1/truth_dev.ck"
TEST_SET="data/NCBI-oursnew1/truth_test.ck"

MODEL_ROOT=./models/$MODEL_NAME
TRAINING_SET=$MODEL_ROOT/annotations.ck

if [ DEV_SET == "" ]; then
    DEV_SET=$TRAINING_SET
fi

if [ TEST_SET == "" ]; then
    TEST_SET=$TRAINING_SET
fi

mkdir -p $MODEL_ROOT/encoded_data

if [ $MUST_RE_RUN == 1 ] || [ ! -e $MODEL_ROOT/encoded_data/test.pk ]; then
    echo ${green}=== Encoding Dataset ===${reset}
    python preprocess_partial_ner/encode_folder.py --input_train $TRAINING_SET --input_testa $DEV_SET --input_testb $TEST_SET --pre_word_emb data/glove.200.pk --output_folder $MODEL_ROOT/encoded_data/
fi

CHECKPOINT_DIR=$MODEL_ROOT/checkpoint/
CHECKPOINT_NAME=autoner

echo ${green}=== Training AutoNER Model ===${reset}
python train_partial_ner.py \
    --cp_root $CHECKPOINT_DIR \
    --checkpoint_name $CHECKPOINT_NAME \
    --eval_dataset $MODEL_ROOT/encoded_data/test.pk \
    --train_dataset $MODEL_ROOT/encoded_data/train_0.pk \
    --update SGD --lr 0.05 --hid_dim 300 --droprate 0.5 \
    --sample_ratio 1.0 --word_dim 200 --epoch 50

echo ${green}Done.${reset}
