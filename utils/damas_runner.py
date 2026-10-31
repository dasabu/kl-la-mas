import numpy as np
from algorithms import UCB1_multiple_C, MultiAgentSystem_BO
from environment import DynamicEnvOrdered, Env
from bayesian_optimization import (
    PillarHolder, GPWithPillarReRun2, RemoveOldestNonPillarClass
)


def pre_train_agent_BO(agent, env, n_steps, env_prob=1.0):
    for _ in range(n_steps):
        action = agent.select_action()
        delay = env.sample(action)
        reward = -delay
        forced_p_env = [1.0/agent.n_envs]*agent.n_envs if hasattr(agent, "n_envs") else None
        agent.update(action, reward, delay=delay, env_prob=env_prob, p_env=forced_p_env)
        env.step()
    return agent

def do_run_BO_dataset(env, multi_agent_system, n_steps,
                      pre_train_steps=100,
                      pretrain_mus=None, pretrain_sigmas=None):
    """
    Run DAMAS-BO on a DynamicEnvOrdered env (built from *all* dataset envs),
    while pre-training its fixed agents on a subset (pretrain_mus/sigmas).

    pretrain_mus/sigmas: shape [n_pretrain_envs, num_actions]
                         (e.g., first 3 dataset envs)
    """
    assert pretrain_mus is not None and pretrain_sigmas is not None, \
        "Provide pretrain_mus and pretrain_sigmas."

    n_pre = pretrain_mus.shape[0]
    num_actions = pretrain_mus.shape[1]

    # Static envs just for pre-training (subset)
    static_envs = [Env(mus=pretrain_mus[i], sigmas=pretrain_sigmas[i]) for i in range(n_pre)]

    # Pre-train each fixed agent on its dedicated static env
    for i, agent in enumerate(multi_agent_system.agents):
        # If you pre-train with fewer agents than n_pre, slice;
        # if you have more agents than n_pre, wrap around.
        env_idx = i % n_pre
        pre_train_agent_BO(agent, static_envs[env_idx], n_steps=pre_train_steps, env_prob=1.0)

    # Logs
    selected_actions, rewards, res_delays = [], [], []
    avg_response_times, cum_response_time = [], []
    best_chosen = []
    regret_list, cum_regret, avg_cum_regret = [], [], []
    cumulative_regret = 0.0
    
    env_keys = []
    correct_identifications = []
    correct_id_count = 0
    correct_identifications.append(0.0)
    for step in range(n_steps):
        # Each sub-agent proposes; choose among them with p_env
        actions_per_agent = multi_agent_system.select_action()
        final_action = int(np.random.choice(actions_per_agent, p=multi_agent_system.p_env))

        best_action = env.best_action()
        best_chosen.append(final_action == best_action)
        
        predicted_env = int(np.argmax(multi_agent_system.p_env))
        actual_env = env.current_env()
        env_keys.append(actual_env)
        if predicted_env == actual_env:
            correct_id_count += 1
        correct_identifications.append(correct_id_count / (step + 1))

        delay = env.sample(final_action)
        best_delay = env.sample(best_action)
        reward = -delay

        inst_regret = delay - best_delay
        regret_list.append(inst_regret)
        cumulative_regret += inst_regret
        cum_regret.append(cumulative_regret)
        avg_cum_regret.append(cumulative_regret / (step + 1))

        env.step()

        selected_actions.append(final_action)
        rewards.append(reward)
        res_delays.append(delay)
        avg_delay = float(np.mean(res_delays))
        avg_response_times.append(avg_delay)
        cum_response_time.append(float(np.sum(res_delays)))

        multi_agent_system.track_response_time(delay)
        multi_agent_system.update_q_values(final_action, reward, delay)
        multi_agent_system.update_environment_probabilities(final_action, reward)

    return {
        "rewards": rewards,
        "cum_response_time": cum_response_time,
        "average_response_time": avg_response_times,
        "res": res_delays,
        "best_chosen": [np.mean(best_chosen[:t+1]) for t in range(len(best_chosen))],
        "cumulative_regret": cum_regret,
        "average_cumulative_regret": avg_cum_regret,
        "correct_identifications": correct_identifications
    }

def run_damas_single_run(mus_all, sigmas_all, num_actions, n_steps, change_every,
                         pre_train_steps=100, n_envs=3, verbose=False):
    """
    n_envs: how many dataset envs to use for DAMAS pre-training (e.g., 3).
            Online execution still rotates through *all* rows in mus_all/sigmas_all.
    """
    # Online env cycles through all dataset envs (e.g., 4 CSV files)
    env_damas = DynamicEnvOrdered(mus_all, sigmas_all, change_every)

    # BO helpers for c tuning
    pillar_holder = PillarHolder(num_pillar_points=8)
    gp = GPWithPillarReRun2(pillar_holder)
    remove_oldest = RemoveOldestNonPillarClass(pillar_holder)

    c_grid = np.linspace(0.01, 10.0, 10)
    agent_params = {
        "n_envs": n_envs,  # fixed # of DAMAS agents
        "c_list": c_grid,
        "n_actions": num_actions,
        "c_timestep_change": 40,
        "add_points_functions": [gp.select_points_GP for _ in range(n_envs)],
        "remove_points_function": remove_oldest.remove_oldest_non_pillar,
        "verbose": verbose,
        "c_min": 0.01,
        "c_max": 10.0,
        "var_window": 50
    }
    damas = MultiAgentSystem_BO(UCB1_multiple_C, agent_params, n_envs=n_envs)

    # Pre-train on the *first* n_envs dataset rows (reorder file_names if needed)
    pre_mus = mus_all[:n_envs]
    pre_sig = sigmas_all[:n_envs]

    out = do_run_BO_dataset(
        env=env_damas,
        multi_agent_system=damas,
        n_steps=n_steps,
        pre_train_steps=pre_train_steps,
        pretrain_mus=pre_mus,
        pretrain_sigmas=pre_sig
    )

    avg_resp  = np.array(out["average_response_time"])
    avg_regret = np.array(out["average_cumulative_regret"])
    pbest    = np.array(out["best_chosen"])
    # For DAMAS (3 agents vs 4 true envs)
    env_id_acc = np.full(n_steps, np.nan)
    env_id_acc_series = np.array(out["correct_identifications"])
    if len(env_id_acc_series) == n_steps + 1:
        env_id_acc = env_id_acc_series[1:]
    else:
        env_id_acc = env_id_acc_series[:n_steps]
    return avg_resp, avg_regret, pbest, env_id_acc
