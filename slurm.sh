#!/bin/bash
# This is a template for a simple SLURM sbatch job script file
#
# First, SBATCH options - these could be passed via the command line, but this
# is more convenient
#
#SBATCH --job-name="GPU Assignment Test" #Name of the job which appears in squeue
#
#SBATCH --mail-type=ALL #What notifications are sent by email
#SBATCH --mail-user=ruffner@cs.virginia.edu
#
# Set up your user environment!!
#SBATCH --get-user-env
#
#SBATCH --error="my_job.err"                    # Where to write std err
#SBATCH --output="my_job.output"                # Where to write stdout

./vector_max 1600000 -k 1
./vector_max 1600000 -k 2
./vector_max 1600000 -k 3
./vector_max 1600000 -k 4



#./2d_convolve 400 100 -k 1
