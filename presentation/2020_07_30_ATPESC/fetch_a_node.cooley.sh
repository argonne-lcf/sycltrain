#!/bin/bash  -x

#qsub -I -n 1 -q debug -A Performance -t 120
qsub -I -n 1 -q training -A ATPESC2020 -t 120
