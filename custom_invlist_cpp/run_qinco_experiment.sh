set -e


function run_on ()
{
    sys="$1"
    shift
    name="$1"
    shift
    script="$logdir/$name.sh"

    if [ -e "$script" ]; then
        echo script "$script" exists
        return
    fi

    # srun handles special characters fine, but the shell interpreter
    # does not
    escaped_cmd=$( printf "%q " "$@" )

    cat > $script <<EOF
#! /bin/bash
srun $escaped_cmd
EOF

    echo -n "$logdir/$name.stdout "
    sbatch -n1 -J "$name" \
           $sys \
            --comment='priority is the only one that works'  \
           --output="$logdir/$name.stdout" \
           "$script"

}

# see https://fb.workplace.com/groups/airesearchinfrausers/posts/2560056980817532/?comment_id=2565685580254672
function run_on_1machine_for_timing {
    run_on "--cpus-per-task=80 --gres=gpu:1  --mem=480G --time=70:00:00 --constraint=bldg1 --partition=learnlab" "$@"
}


###############################
# To enable / disable some experiments
###############################

function SKIP () {
    echo -n
}

function RUN () {
    "$@"
}

datadir=/checkpoint/matthijs/id_compression/qinco/
logdir=$datadir/logs
mkdir -p $logdir 

##########################################
# 10M scale experiment
##########################################

if false; then



tmpdir=/scratch/matthijs/

# data from https://github.com/facebookresearch/Qinco/blob/main/docs/IVF_search.md

(cd $tmpdir; wget https://dl.fbaipublicfiles.com/QINCo/models/bigann_IVF65k_16x8_L2.pt )
(cd $tmpdir ; wget https://dl.fbaipublicfiles.com/QINCo/ivf/bigann10M_IVF65k_16x8_L2.faissindex )

# run baseline without id compression
# parameters are one of the optimal op points from 
# https://gist.github.com/mdouze/e4b7c9dbf6a52e0f7cf100ce0096aaa8
# cno=21

#  baseline 

python search_ivf.py --db bigann10M \
 --model $tmpdir/bigann_IVF65k_16x8_L2.pt \
 --index $tmpdir/bigann10M_IVF65k_16x8_L2.faissindex \
 --todo search  --nthread 32 --nprobe 64 --nshort 100 

# with ROC 

python search_ivf.py --db bigann10M \
 --model $tmpdir/bigann_IVF65k_16x8_L2.pt \
 --index $tmpdir/bigann10M_IVF65k_16x8_L2.faissindex \
 --todo search  --nthread 32 --nprobe 64 --nshort 100  \
 --id_compression roc --defer_id_decoding



##########################################
# 1B scale experiment 8x8
##########################################


(cd $datadir && wget https://dl.fbaipublicfiles.com/QINCo/ivf/bigann1B_IVF1M_8x8_L2.faissindex)
(cd $datadir && wget https://dl.fbaipublicfiles.com/QINCo/models/bigann_IVF1M_8x8_L2.pt)

fi 

# cno=533 from https://gist.github.com/mdouze/0187c2ca3f96f806e41567af13f80442
# fastest run with R@1 > 0.3
params="--nprobe 128 --quantizer_efSearch 64 --nshort 200"

SKIP run_on_1machine_for_timing BL.8x8_L2.a \
    python -u search_ivf_qinco.py \
      --todo search \
      --db bigann1B \
      --model $datadir/bigann_IVF1M_8x8_L2.pt \
      --index $datadir/bigann1B_IVF1M_8x8_L2.faissindex \
      --nthread 32 \
      $params

#.c: defer id decoding
#.d: redo several times to stabilize timings 
for comp in none packed-bits elias-fano roc wavelet-tree wavelet-tree-1; do 

    run_on_1machine_for_timing ICOMP.8x8_L2.$comp.d \
        python -u search_ivf_qinco.py \
        --todo search \
        --db bigann1B \
        --model $datadir/bigann_IVF1M_8x8_L2.pt \
        --index $datadir/bigann1B_IVF1M_8x8_L2.faissindex \
        --nthread 32 \
        --id_compression $comp --defer_id_decoding --redo_search 10 \
        $params

done 