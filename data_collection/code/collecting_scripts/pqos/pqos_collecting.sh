#!/bin/bash
current=$(date "+%Y%m%d%H%M%S")
source /usr/share/miniconda/etc/profile.d/conda.sh
conda activate base 
python /home/fsp/czy/RDT/monitoring.py 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53 $1 > $current-pqos-results.log

