#!/bin/bash

#PBS -l select=1:ncpus=32:mem=2gb -l place=pack

#PBS -l walltime=3:00:00

#PBS -q short_cpuQ

module load mpich-3.2

mpirun.actual -n 32 ./huffman_encoding file_512M.txt
