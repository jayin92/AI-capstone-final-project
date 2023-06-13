#!/bin/bash

# Run the ACO algorithm for the TSP problem
problem_list="berlin52 rat99 bier127 ch130 a280"

for val in $problem_list; do
    python ACO/main.py --problem $val --num_ants 25 --plots --num_workers 8 --max_iter 200 --patience 1000
    python ACO/main.py --problem $val --num_ants 25 --plots --num_workers 8 --max_iter 200 --patience 1000
    python ACO/main.py --problem $val --num_ants 25 --plots --num_workers 8 --max_iter 200 --patience 1000
    python ACO/main.py --problem $val --num_ants 25 --plots --num_workers 8 --max_iter 200 --patience 1000
    python ACO/main.py --problem $val --num_ants 25 --plots --num_workers 8 --max_iter 200 --patience 1000
done