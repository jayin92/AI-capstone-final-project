import argparse
import time
import os
import matplotlib.pyplot as plt

from tsp import TSP
from mmas import MMAS

problem_list = [
    "berlin52",
    "rat99",
    "bier127",
    "ch130",
    "a280",
]

if __name__ == "__main__":
    plt.ion()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--problem", 
        type=str, 
        default="berlin52", 
        choices=problem_list,
        help="choose problem")

    parser.add_argument(
        "--num_ants",
        type=int,
        default=10,
        help="number of ants"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="number of workers"
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="plot the results"
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=2000,
        help="maximum number of iterations"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="patience"
    )
    args = parser.parse_args()
    problem_name = args.problem
    para = {
        "problem_name": problem_name,
        "num_ants": args.num_ants,
        "num_workers": args.num_workers,
        "plots": args.plots,
        "max_iter": args.max_iter,
        "patience": args.patience,
        "alpha": 1,
        "beta": 2,
        "rho": 0.2,
        "q": 1,
        "name": "MMAS"
    }
    tsp = eval(para["name"])(**para)
    start_time = time.time()
    res = tsp.run()
    print("Time taken: ", time.time() - start_time)
    print("Best cost: ", res[1])
    # Create dir
    os.makedirs(f"./logs/{para['name']}", exist_ok=True)
    with open(f"./logs/{para['name']}/{para['problem_name']}.txt", "a") as f:
        f.write("--------------------\n")
        # Save configuration
        f.write("Problem: {}\n".format(problem_name))
        f.write(f"Configurations: {str(para)}\n")
        # Save results
        f.write("Best path: {}\n".format(res[0]))
        f.write("Best cost: {}\n".format(res[1]))
        f.write("Time taken: {}\n".format(time.time() - start_time))
