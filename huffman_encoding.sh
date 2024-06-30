#!/bin/bash

#PBS -l select=1:ncpus=4:mem=2gb -l place=scatter

#PBS -l walltime=0:10:00

#PBS -q short_cpuQ

module load mpich-3.2
mpirun.actual -n 1 ./huffman_encoding random_text_1gb.txt
mpirun.actual -n 1 ./huffman_encoding random_text_1gb.txt
mpirun.actual -n 1 ./huffman_encoding random_text_1gb.txt
mpirun.actual -n 1 ./huffman_encoding random_text_1gb.txt
mpirun.actual -n 1 ./huffman_encoding random_text_1gb.txt
mpirun.actual -n 1 ./huffman_encoding random_text_1gb.txt
mpirun.actual -n 1 ./huffman_encoding random_text_1gb.txt
mpirun.actual -n 1 ./huffman_encoding random_text_1gb.txt
mpirun.actual -n 1 ./huffman_encoding random_text_1gb.txt
mpirun.actual -n 1 ./huffman_encoding random_text_1gb.txt



