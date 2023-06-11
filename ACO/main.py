import argparse
import matplotlib.pyplot as plt


from tsp import TSP

problem_list = [
    "berlin52",
    "rat99",
    "bier127",
    "ch130",
    "a280",
   # "d657",
   # "fl3795",
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
    args = parser.parse_args()
    problem_name = args.problem
    tsp = TSP(problem_name, 
              alpha=0.9, 
              beta=6, 
              max_iter=2000, 
              num_ants=args.num_ants, 
              patience=5, 
              rho=0.05, 
              q=100, 
              num_workers=args.num_workers, 
              plots=args.plots
              )
    res = tsp.run()

