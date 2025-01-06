#!/bin/bash

############# optimized version with unroll (uncomment unroll in main.cu first) #############
# Define output file
OUTPUT_FILE="./optimized_unrolled_16.txt"

# Clear output file if it exists
> $OUTPUT_FILE

# Define parameters
PARTITION="exercise-gpu"
GPU_RESOURCE="gpu:1"
BINARY="./bin/nbody"
BLOCK_DIMENSIONS=1024
NUM_ITERATIONS=100


# Array of problem sizes to test
PROBLEM_SIZE=(2 4 8 16 32 64 128 265 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576)

# Loop through problem sizeses
for SIZE in "${PROBLEM_SIZE[@]}"; do
  #echo "Running with PROBLEMSIZE=${SIZE}" | tee -a $OUTPUT_FILE
  srun --partition=$PARTITION --gres=$GPU_RESOURCE $BINARY -s $SIZE -i $NUM_ITERATIONS -t $BLOCK_DIMENSIONS --silent --shared >> $OUTPUT_FILE 2>&1
  echo "----------------------------------------------------" >> $OUTPUT_FILE
done

echo " Optimized Benchmark completed. Results saved to $OUTPUT_FILE."