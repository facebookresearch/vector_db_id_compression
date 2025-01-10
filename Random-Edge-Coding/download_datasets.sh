#!/bin/bash

[[ -z "$DATA_DIR" ]] && { echo "Please set DATA_DIR to something" ; exit 1; }

REC_DATA_DIR=$DATA_DIR/rec
mkdir -p $REC_DATA_DIR
cd $REC_DATA_DIR

# social media
wget http://konect.cc/files/download.tsv.youtube-u-growth.tar.bz2
wget https://nrvis.com/download/data/soc/soc-FourSquare.zip
wget https://nrvis.com/download/data/soc/soc-digg.zip
wget http://snap.stanford.edu/data/loc-gowalla_edges.txt.gz 

# non-social media
wget http://snap.stanford.edu/data/as-skitter.txt.gz
wget http://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz

# extract
tar -xf download.tsv.youtube-u-growth.tar.bz2
unzip -o soc-FourSquare.zip
unzip -o soc-digg.zip
gunzip loc-gowalla_edges.txt.gz
gunzip com-dblp.ungraph.txt.gz
gunzip as-skitter.txt.gz

cd -
