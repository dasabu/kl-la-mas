import pandas as pd
from environment import real_world_workloads
from utils import run_methods


def main():
    mus_all, sigmas_all = real_world_workloads()
    n_steps = 4000
    change_every = 1000
    runs = 2

    results, changepoint_results = run_methods(
        mus_all, sigmas_all,
        n_steps=n_steps,
        change_every=change_every,
        runs=runs,
        include_damas=True,
        damas_pre_train_steps=100,
        damas_n_envs=2
    )

    output_csv = 'aggregated_results.csv'
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f'Aggregated results written to {output_csv}')

    changepoint_file = 'changepoint_results.csv'
    pd.DataFrame(changepoint_results).to_csv(changepoint_file, index=False)
    print(f"Changepoint results saved to {changepoint_file}")


if __name__ == "__main__":
    main()
