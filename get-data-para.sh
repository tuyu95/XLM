# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

set -e

pair=$1  # input language pair

# data paths
MAIN_PATH=$PWD
PARA_PATH=$PWD/data/para
PROCESSED_PATH=$PWD/data/processed/XLM
# CODES_PATH=$MAIN_PATH/guen.bpe
# VOCAB_PATH=$MAIN_PATH/vocab.guen
# CODES_PATH=$MAIN_PATH/codes_xnli_15
# VOCAB_PATH=$MAIN_PATH/vocab_xnli_15

# tools paths
TOOLS_PATH=$PWD/tools
TOKENIZE=$TOOLS_PATH/tokenize.sh
LOWER_REMOVE_ACCENT=$TOOLS_PATH/lowercase_and_remove_accent.py
FASTBPE=$TOOLS_PATH/fastBPE/fast

# install tools
./install-tools.sh

# create directories
mkdir -p $PARA_PATH
mkdir -p $PROCESSED_PATH


#
# Download and uncompress data
#
# en-gu
if [ ! -f $PARA_PATH/bible.gu-en.tsv ];
then
  wget http://data.statmt.org/wmt19/translation-task/bible.gu-en.tsv.gz -O $PARA_PATH/bible.gu-en.tsv.gz
  gzip -d $PARA_PATH/bible.gu-en.tsv.gz
  awk -F$'\t' '{print $1}' $PARA_PATH/bible.gu-en.tsv > $PARA_PATH/bi.gu-en.gu
  awk -F$'\t' '{print $2}' $PARA_PATH/bible.gu-en.tsv > $PARA_PATH/bi.gu-en.en
fi

if [ ! -f $PARA_PATH/govin-clean.gu-en.tsv ];
then
  wget http://data.statmt.org/wmt19/translation-task/govin-clean.gu-en.tsv.gz -O $PARA_PATH/govin-clean.gu-en.tsv.gz
  gzip -d $PARA_PATH/govin-clean.gu-en.tsv.gz
  awk -F$'\t' '{print $1}' $PARA_PATH/govin-clean.gu-en.tsv > $PARA_PATH/govin-clean.gu-en.gu
  awk -F$'\t' '{print $2}' $PARA_PATH/govin-clean.gu-en.tsv > $PARA_PATH/govin-clean.gu-en.en
fi

if [ ! -f $PARA_PATH/wikipedia.gu-en.tsv ];
then
  wget http://data.statmt.org/wmt19/translation-task/wikipedia.gu-en.tsv.gz -O $PARA_PATH/wikipedia.gu-en.tsv.gz
  gzip -d $PARA_PATH/wikipedia.gu-en.tsv.gz
  awk -F$'\t' '{print $1}' $PARA_PATH/wikipedia.gu-en.tsv > $PARA_PATH/wikipedia.gu-en.gu
  awk -F$'\t' '{print $2}' $PARA_PATH/wikipedia.gu-en.tsv > $PARA_PATH/wikipedia.gu-en.en
fi

if [ ! -f $PARA_PATH/opus.gu-en.tsv ];
then
  wget http://data.statmt.org/wmt19/translation-task/opus.gu-en.tsv.gz -O $PARA_PATH/opus.gu-en.tsv.gz
  gzip -d $PARA_PATH/opus.gu-en.tsv.gz
  awk -F$'\t' '{print $1}' $PARA_PATH/opus.gu-en.tsv > $PARA_PATH/opus.gu-en.gu
  awk -F$'\t' '{print $2}' $PARA_PATH/opus.gu-en.tsv > $PARA_PATH/opus.gu-en.en
fi

if [ $pair == "en-gu" ]; then
  cat $PARA_PATH/bi.gu-en.gu $PARA_PATH/govin-clean.gu-en.gu $PARA_PATH/wikipedia.gu-en.gu $PARA_PATH/opus.gu-en.gu > $PARA_PATH/comb.en-gu.gu
  cat $PARA_PATH/bi.gu-en.en $PARA_PATH/govin-clean.gu-en.en $PARA_PATH/wikipedia.gu-en.en $PARA_PATH/opus.gu-en.en > $PARA_PATH/comb.en-gu.en
fi


# ar-en
if [ $pair == "ar-en" ]; then
  # OpenSubtitles 2018
  # wget -c http://opus.nlpl.eu/download.php?f=OpenSubtitles2018%2Far-en.txt.zip -P $PARA_PATH
  # MultiUN
  wget -c http://opus.nlpl.eu/download.php?f=MultiUN%2Far-en.txt.zip -P $PARA_PATH
  unzip -u $PARA_PATH/download.php?f=MultiUN%2Far-en.txt.zip -d $PARA_PATH
fi

# bg-en
if [ $pair == "bg-en" ]; then
  # OpenSubtitles 2018
  # wget -c http://opus.nlpl.eu/download.php?f=OpenSubtitles2018%2Fbg-en.txt.zip -P $PARA_PATH
  # EU Bookshop
  wget -c http://opus.nlpl.eu/download.php?f=EUbookshop%2Fbg-en.txt.zip -P $PARA_PATH
  unzip -u $PARA_PATH/download.php?f=EUbookshop%2Fbg-en.txt.zip -d $PARA_PATH
  # Europarl
  wget -c http://opus.nlpl.eu/download.php?f=Europarl%2Fbg-en.txt.zip -P $PARA_PATH
  unzip -u $PARA_PATH/download.php?f=Europarl%2Fbg-en.txt.zip -d $PARA_PATH
fi

# de-en
if [ $pair == "de-en" ]; then
  # OpenSubtitles 2018
  # wget -c http://opus.nlpl.eu/download.php?f=OpenSubtitles2018%2Fde-en.txt.zip -P $PARA_PATH
  # EU Bookshop
  wget -c http://opus.nlpl.eu/download.php?f=EUbookshop%2Fde-en.txt.zip -P $PARA_PATH
  unzip -u $PARA_PATH/download.php?f=EUbookshop%2Fde-en.txt.zip -d $PARA_PATH
fi

# el-en
if [ $pair == "el-en" ]; then
  # OpenSubtitles 2018
  # wget -c http://opus.nlpl.eu/download.php?f=OpenSubtitles2018%2Fel-en.txt.zip -P $PARA_PATH
  # EU Bookshop
  wget -c http://opus.nlpl.eu/download.php?f=EUbookshop%2Fel-en.txt.zip -P $PARA_PATH
  unzip -u $PARA_PATH/download.php?f=EUbookshop%2Fel-en.txt.zip -d $PARA_PATH
fi

# en-es
if [ $pair == "en-es" ]; then
  # OpenSubtitles 2018
  # wget -c http://opus.nlpl.eu/download.php?f=OpenSubtitles2018%2Fen-es.txt.zip -P $PARA_PATH
  # EU Bookshop
  # wget -c http://opus.nlpl.eu/download.php?f=EUbookshop%2Fen-es.txt.zip -P $PARA_PATH
  # MultiUN
  wget -c https://object.pouta.csc.fi/OPUS-MultiUN/v1/moses/en-es.txt.zip -P $PARA_PATH
  unzip -u $PARA_PATH/en-es.txt.zip -d $PARA_PATH
fi

# en-fr
if [ $pair == "en-fr" ]; then
  echo "Download parallel data for English-Hindi"
  # OpenSubtitles 2018
  # wget -c http://opus.nlpl.eu/download.php?f=OpenSubtitles2018%2Fen-fr.txt.zip -P $PARA_PATH
  # EU Bookshop
  # wget -c http://opus.nlpl.eu/download.php?f=EUbookshop%2Fen-fr.txt.zip -P $PARA_PATH
  # MultiUN
  wget -c https://object.pouta.csc.fi/OPUS-MultiUN/v1/moses/en-fr.txt.zip -P $PARA_PATH
  unzip -u $PARA_PATH/en-fr.txt.zip -d $PARA_PATH
fi

# en-hi
if [ $pair == "en-hi" ]; then
  echo "Download parallel data for English-Hindi"
  # IIT Bombay English-Hindi Parallel Corpus
  wget -c http://www.cfilt.iitb.ac.in/iitb_parallel/iitb_corpus_download/parallel.tgz -P $PARA_PATH
  tar -xvf $PARA_PATH/parallel.tgz -d $PARA_PATH
fi

# en-ru
if [ $pair == "en-ru" ]; then
  echo "Download parallel data for English-Russian"
  # OpenSubtitles 2018
  # wget -c http://opus.nlpl.eu/download.php?f=OpenSubtitles2018%2Fen-ru.txt.zip -P $PARA_PATH
  # MultiUN
  wget -c http://opus.nlpl.eu/download.php?f=MultiUN%2Fen-ru.txt.zip -P $PARA_PATH
  unzip -u download.php?f=MultiUN%2Fen-ru.txt.zip -d $PARA_PATH
fi

# en-sw
if [ $pair == "en-sw" ]; then
  echo "Download parallel data for English-Swahili"
  # Tanzil
  wget -c http://opus.nlpl.eu/download.php?f=Tanzil%2Fen-sw.txt.zip -P $PARA_PATH
  unzip -u download.php?f=Tanzil%2Fen-sw.txt.zip -d $PARA_PATH
  # GlobalVoices
  wget -c http://opus.nlpl.eu/download.php?f=GlobalVoices%2Fen-sw.txt.zip -P $PARA_PATH
  unzip -u download.php?f=GlobalVoices%2Fen-sw.txt.zip -d $PARA_PATH
fi

# en-th
if [ $pair == "en-th" ]; then
  echo "Download parallel data for English-Thai"
  # OpenSubtitles 2018
  wget -c http://opus.nlpl.eu/download.php?f=OpenSubtitles2018%2Fen-th.txt.zip -P $PARA_PATH
  unzip -u $PARA_PATH/download.php?f=OpenSubtitles2018%2Fen-th.txt.zip -d $PARA_PATH
fi

# en-tr
if [ $pair == "en-tr" ]; then
  echo "Download parallel data for English-Turkish"
  # OpenSubtitles 2018
  # wget -c http://opus.nlpl.eu/download.php?f=OpenSubtitles2018%2Fen-tr.txt.zip -P $PARA_PATH
  # SETIMES2
  wget -c http://opus.nlpl.eu/download.php?f=SETIMES2%2Fen-tr.txt.zip -P $PARA_PATH
  unzip -u $PARA_PATH/download.php?f=SETIMES2%2Fen-tr.txt.zip -d $PARA_PATH
  # Wikipedia
  wget -c http://opus.nlpl.eu/download.php?f=Wikipedia%2Fen-tr.txt.zip -P $PARA_PATH
  unzip -u $PARA_PATH/download.php?f=Wikipedia%2Fen-tr.txt.zip -d $PARA_PATH
  # TED
  wget -c https://object.pouta.csc.fi/OPUS-TED2013/v1.1/moses/en-tr.txt.zip -P $PARA_PATH
  unzip -u $PARA_PATH/en-tr.txt.zip -d $PARA_PATH
fi

# en-ur
if [ $pair == "en-ur" ]; then
  echo "Download parallel data for English-Urdu"
  # OpenSubtitles 2018
  # wget -c http://opus.nlpl.eu/download.php?f=OpenSubtitles2018%2Fen-ur.txt.zip -P $PARA_PATH
  # Tanzil
  wget -c http://opus.nlpl.eu/download.php?f=Tanzil%2Fen-ur.txt.zip -P $PARA_PATH
  unzip -u $PARA_PATH/download.php?f=Tanzil%2Fen-ur.txt.zip -d $PARA_PATH
fi

# en-vi
if [ $pair == "en-vi" ]; then
  echo "Download parallel data for English-Vietnamese"
  # OpenSubtitles 2018
  wget -c http://opus.nlpl.eu/download.php?f=OpenSubtitles2018%2Fen-vi.txt.zip -P $PARA_PATH
  unzip -u $PARA_PATH/download.php?f=OpenSubtitles2018%2Fen-vi.txt.zip -d $PARA_PATH
fi

# en-zh
if [ $pair == "en-zh" ]; then
  echo "Download parallel data for English-Chinese"
  # OpenSubtitles 2016
  # wget -c http://opus.nlpl.eu/download.php?f=OpenSubtitles2016%2Fen-zh.txt.zip -P $PARA_PATH
  # MultiUN
  wget -c http://opus.nlpl.eu/download.php?f=MultiUN%2Fen-zh.txt.zip -P $PARA_PATH
  unzip -u $PARA_PATH/download.php?f=MultiUN%2Fen-zh.txt.zip -d $PARA_PATH
fi

# en-gu
if [ $pair == "en-hi" ]; then
  echo "Download parallel data for EN-GU"
  # IIT Bombay English-Hindi Parallel Corpus
  wget -c http://data.statmt.org/wmt19/translation-task/bible.gu-en.tsv.gz -P $PARA_PATH
  # tar -xvf $PARA_PATH/parallel.tgz -d $PARA_PATH
fi

#
# Tokenize and preprocess data
#

# tokenize
for lg in $(echo $pair | sed -e 's/\-/ /g'); do
  if [ ! -f $PARA_PATH/$pair.$lg.all ]; then
    cat $PARA_PATH/*.$pair.$lg | $TOKENIZE $lg | python $LOWER_REMOVE_ACCENT > $PARA_PATH/$pair.$lg.all
  fi
done

# train en-gu bpe code and vocab
# cat $PARA_PATH/en-gu.en.all $PARA_PATH/en-gu.gu.all $MAIN_PATH/data/mono/en/all.en.tok $MAIN_PATH/data/mono/gu/all.gu.tok > $PARA_PATH/en-gu.engu.all
# $FASTBPE learnbpe 15000 $PARA_PATH/en-gu.en.all $PARA_PATH/en-gu.gu.all > $MAIN_PATH/codes_engu
# $CODES_PATH=$MAIN_PATH/codes_engu
# $FASTBPE applybpe $PARA_PATH/en-gu.engu.all.15000 $PARA_PATH/en-gu.engu.all $MAIN_PATH/codes_engu
# $FASTBPE getvocab $PARA_PATH/en-gu.engu.all.15000> $MAIN_PATH/vocab_engu
# $VOCAB_PATH=$MAIN_PATH/vocab_engu

# split into train / valid / test
split_data() {
    get_seeded_random() {
        seed="$1"; openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt </dev/zero 2>/dev/null
    };
    NLINES=`wc -l $1  | awk -F " " '{print $1}'`;
    NTRAIN=$((NLINES - 10000));
    NVAL=$((NTRAIN + 5000));
    echo $NLINES
    echo $NTRAIN
    echo $NVAL
    shuf --random-source=<(get_seeded_random 42) $1 | head -$NTRAIN             > $2;
    shuf --random-source=<(get_seeded_random 42) $1 | head -$NVAL | tail -5000  > $3;
    shuf --random-source=<(get_seeded_random 42) $1 | tail -5000                > $4;
    echo "shuf done"
}
for lg in $(echo $pair | sed -e 's/\-/ /g'); do
  split_data $PARA_PATH/$pair.$lg.all $PARA_PATH/$pair.$lg.train $PARA_PATH/$pair.$lg.valid $PARA_PATH/$pair.$lg.test
  echo "split data"
done

# Get BPE codes and vocab
# wget -c https://dl.fbaipublicfiles.com/XLM/codes_xnli_15 -P $MAIN_PATH
# wget -c https://dl.fbaipublicfiles.com/XLM/vocab_xnli_15 -P $MAIN_PATH

# apply BPE codes and binarize the parallel corpora
for lg in $(echo $pair | sed -e 's/\-/ /g'); do
  for split in train valid test; do
    $FASTBPE applybpe $PROCESSED_PATH/$pair.$lg.$split $PARA_PATH/$pair.$lg.$split $MAIN_PATH/codes
    python preprocess.py $MAIN_PATH/vocab.en-gu $PROCESSED_PATH/$pair.$lg.$split
  done
done
