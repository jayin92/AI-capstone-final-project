import argparse
import matplotlib.pyplot as plt


from tsp import TSP

problem_list = [
    "berlin52",
    "rat99",
    "a280",
    "d657",
    "fl3795"
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
    args = parser.parse_args()
    problem_name = args.problem
    tsp = TSP(problem_name, alpha=1, beta=1.5, max_iter=100000, num_ants=5, patience=1000, rho=0.1, q=100)
    res = tsp.run()

