# Parallel-Huffman-HPC
Project repository for High Performance Computing for Data Science course UNITN'

### Repository Structure

```plaintext
project_root/
│
├── figures/                 # Directory for figures
├── LICENSE                  # Licensing information
├── README.md                # Introduction and instructions
├── generate.sh              # Shell script used for generating synthetic datasets 
├── huffman_encoding.c       # C source code for parallel Huffman encoding
├── huffman_encoding.sh      # Shell script for Huffman encoding tasks
├── file_10M.txt             # Sample 10 MB text file
├── parallel_huffman.pdf     # Project report document
│
```
## Getting Started
Follow these steps to set up and run the project:

- **Generate Synthetic Datasets**
  The `generate.sh` script generates synthetic datasets using `file_10M.txt` (10 MB) to create datasets ranging from 256 MB to 16384 MB for faster generation
  ```
  chmod +x generate.sh
  ./generate.sh
  ```
- **Module Load**
  ```
  module load mpich-3.2
  ```
- **Code Compilation**
  ```
  mpicc -g -Wall -o huffman_encoding huffman_encoding.c
  ```
- **PBS job instruction and submission**

  Set parameters such as the number of chunks, cores, walltime, dataset size, etc in `huffman_encoding.sh`, and submit the job using the following command. 
  ```
    qsub huffman_encoding.sh
  ```

## Results

**Runtime difference**

<img src="figures/time_spent.png" alt="Alt Text" width="500" />

**Data distribution and gathering**

<img src="figures/data_distribution.jpg" alt="Alt Text" width="500" />

**Runtime, speedup, efficiency**

Asterisks (*) indicate multiple sends due to integer constraints in MPI.

| ![Runtime](figures/runtime.png) <br> **Figure 3:** Runtime | ![Speedup](figures/speedup.png) <br> **Figure 4:** Speedup | ![Speedup](figures/efficiency.png) <br> **Figure 5:** Efficiency |
|:--:|:--:|:--:|


**Scalability**

| ![Strong scalability](figures/strong_scale.png) <br> **Figure 6:** Strong scalability | ![Weak scalability](figures/weak_scale.png) <br> **Figure 7:** Weak scalability |
|:--:|:--:|






