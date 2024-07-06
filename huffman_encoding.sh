#!/bin/bash

#PBS -l select=1:ncpus=64:mem=2gb -l place=pack

#PBS -l walltime=0:30:00

#PBS -q short_cpuQ

module load mpich-3.2
mpirun.actual -n 1 ./huffman_encoding file_2048M.txt
