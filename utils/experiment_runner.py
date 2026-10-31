import numpy as np
from environment import DynamicEnvOrdered
from algorithms import (
    UCB1, UCB1_with_BO_Optimization,
    LikelihoodAdaptiveMultiAgentSystem,
    GLRklUCB, MonitoredUCB
)
from bayesian_optimization import (
    PillarHolder, GPWithPillarReRun2, RemoveOldestNonPillarClass
)
from .damas_runner import run_damas_single_run


def run_methods(mus_all, sigmas_all, n_steps=4000, change_every=1000, runs=10,
                include_damas=True, damas_pre_train_steps=100, damas_n_envs=3):
    n_env, num_actions = mus_all.shape
    results = []
    
    changepoint_results = []  # New: collect changepoint data
    
    for run in range(runs):
        # Create separate envs for each algorithm
        env_lam = DynamicEnvOrdered(mus_all, sigmas_all, change_every)
        env_lam_bo = DynamicEnvOrdered(mus_all, sigmas_all, change_every)
        env_glr = DynamicEnvOrdered(mus_all, sigmas_all, change_every)
        env_mucb = DynamicEnvOrdered(mus_all, sigmas_all, change_every)

        # LA-MAS (standard)
        lam_params = {'c': 1.0, 'n_actions': num_actions, 'var_window': 50}
        lam = LikelihoodAdaptiveMultiAgentSystem(UCB1, lam_params,
                                                 detection_threshold=0.00005,
                                                 cooldown_steps=200,
                                                 min_data_points=40)

        # LA-MAS-BO with full BO
        pillar_holder = PillarHolder(num_pillar_points=5)
        gp = GPWithPillarReRun2(pillar_holder)
        remove_oldest = RemoveOldestNonPillarClass(pillar_holder)
        c_list = np.linspace(0.1, 5.0, 5)
        agent_params_bo = {
            'add_points_functions': [gp.select_points_GP],
            'remove_points_function': remove_oldest.remove_oldest_non_pillar,
            'c_list': c_list,
            'n_actions': num_actions,
            'var_window': 50,
            'c_timestep_change': 20,
            'c_min': 0.1,
            'c_max': 10.0
        }
        lam_bo = LikelihoodAdaptiveMultiAgentSystem(
            UCB1_with_BO_Optimization,
            agent_params_bo,
            detection_threshold=0.00005,
            delta=0.07,
            cooldown_steps=200,
            min_data_points=40
        )

        # GLR-klUCB
        glr = GLRklUCB(
            n_actions=num_actions,
            sigma=np.mean(sigmas_all),
            glr_window=50,
            glr_threshold=15.0
        )

        # M-UCB
        mucb = MonitoredUCB(
            n_actions=num_actions,
            w=20,
            b=5.0,
            gamma=0.2,
            c=1.0
        )

        # Lists for metrics
        best_frac_lam, best_frac_lam_bo = [], []
        best_frac_glr, best_frac_mucb = [], []
        avg_resp_lam, avg_resp_lam_bo = [], []
        avg_resp_glr, avg_resp_mucb = [], []
        avg_regret_lam, avg_regret_lam_bo = [], []
        avg_regret_glr, avg_regret_mucb = [], []
        env_id_acc_lam, env_id_acc_lam_bo = [0], [0]

        # Cumulative regrets
        cum_regret_lam = 0.0
        cum_regret_lam_bo = 0.0
        cum_regret_glr = 0.0
        cum_regret_mucb = 0.0

        # Track per-step rewards and best-action selections
        rewards_lam, rewards_lam_bo = [], []
        rewards_glr, rewards_mucb = [], []
        best_chosen_lam, best_chosen_lam_bo = [], []
        best_chosen_glr, best_chosen_mucb = [], []

        # For environment ID accuracy
        env_correct_lam = 0
        env_correct_lam_bo = 0
        
        # Generate true changepoints
        true_changepoints = list(range(change_every, n_steps, change_every))


        for step in range(n_steps):
            # LA-MAS
            a_lam = lam.select_action()
            best_action = env_lam.best_action()
            best_chosen_lam.append(a_lam == best_action)
            delay = env_lam.sample(a_lam)
            delay_best = env_lam.sample(best_action)
            inst_regret = delay - delay_best
            cum_regret_lam += inst_regret
            avg_regret_lam.append(cum_regret_lam / (step + 1))
            rewards_lam.append(-delay)
            lam.update(a_lam, -delay, delay)
            if np.argmax(lam.p_env) == env_lam.current_env():
                env_correct_lam += 1
            env_id_acc_lam.append(env_correct_lam / (step + 1))
            env_lam.step()

            # LA-MAS-BO
            a_bo = lam_bo.select_action()
            best_action_bo = env_lam_bo.best_action()
            best_chosen_lam_bo.append(a_bo == best_action_bo)
            delay_bo = env_lam_bo.sample(a_bo)
            delay_best_bo = env_lam_bo.sample(best_action_bo)
            inst_regret_bo = delay_bo - delay_best_bo
            cum_regret_lam_bo += inst_regret_bo
            avg_regret_lam_bo.append(cum_regret_lam_bo / (step + 1))
            rewards_lam_bo.append(-delay_bo)
            lam_bo.update(a_bo, -delay_bo, delay_bo)
            if np.argmax(lam_bo.p_env) == env_lam_bo.current_env():
                env_correct_lam_bo += 1
            env_id_acc_lam_bo.append(env_correct_lam_bo / (step + 1))
            env_lam_bo.step()

            # GLR-klUCB
            a_glr = glr.select_action()
            best_action_glr = env_glr.best_action()
            best_chosen_glr.append(a_glr == best_action_glr)
            delay_glr = env_glr.sample(a_glr)
            delay_best_glr = env_glr.sample(best_action_glr)
            inst_regret_glr = delay_glr - delay_best_glr
            cum_regret_glr += inst_regret_glr
            avg_regret_glr.append(cum_regret_glr / (step + 1))
            rewards_glr.append(-delay_glr)
            glr.update(a_glr, -delay_glr)
            env_glr.step()

            # M-UCB
            a_mucb = mucb.select_action()
            best_action_mucb = env_mucb.best_action()
            best_chosen_mucb.append(a_mucb == best_action_mucb)
            delay_mucb = env_mucb.sample(a_mucb)
            delay_best_mucb = env_mucb.sample(best_action_mucb)
            inst_regret_mucb = delay_mucb - delay_best_mucb
            cum_regret_mucb += inst_regret_mucb
            avg_regret_mucb.append(cum_regret_mucb / (step + 1))
            rewards_mucb.append(-delay_mucb)
            mucb.update(a_mucb, -delay_mucb)
            env_mucb.step()

        # Compute running means of response time and Pbest
        for i in range(n_steps):
            avg_resp_lam.append(sum(-np.array(rewards_lam[:i+1])) / (i+1))
            avg_resp_lam_bo.append(sum(-np.array(rewards_lam_bo[:i+1])) / (i+1))
            avg_resp_glr.append(sum(-np.array(rewards_glr[:i+1])) / (i+1))
            avg_resp_mucb.append(sum(-np.array(rewards_mucb[:i+1])) / (i+1))

            best_frac_lam.append(np.mean(best_chosen_lam[:i+1]))
            best_frac_lam_bo.append(np.mean(best_chosen_lam_bo[:i+1]))
            best_frac_glr.append(np.mean(best_chosen_glr[:i+1]))
            best_frac_mucb.append(np.mean(best_chosen_mucb[:i+1]))

        # Append LA-MAS / LA-MAS-BO / GLR / M-UCB rows
        for i in range(n_steps):
            results.append({'algorithm': 'LA-MAS',     'run': run, 'step': i,
                            'avg_resp': avg_resp_lam[i], 'avg_regret': avg_regret_lam[i],
                            'pbest': best_frac_lam[i], 'env_id_acc': env_id_acc_lam[i]})
            results.append({'algorithm': 'LA-MAS-BO',  'run': run, 'step': i,
                            'avg_resp': avg_resp_lam_bo[i], 'avg_regret': avg_regret_lam_bo[i],
                            'pbest': best_frac_lam_bo[i], 'env_id_acc': env_id_acc_lam_bo[i]})
            results.append({'algorithm': 'GLR',        'run': run, 'step': i,
                            'avg_resp': avg_resp_glr[i], 'avg_regret': avg_regret_glr[i],
                            'pbest': best_frac_glr[i], 'env_id_acc': np.nan})
            results.append({'algorithm': 'M-UCB',      'run': run, 'step': i,
                            'avg_resp': avg_resp_mucb[i], 'avg_regret': avg_regret_mucb[i],
                            'pbest': best_frac_mucb[i], 'env_id_acc': np.nan})

        # ---- DAMAS-BO: run ONCE per run, outside the inner i-loop ----
        if include_damas:
            avg_resp_damas, avg_regret_damas, pbest_damas, env_acc_damas = run_damas_single_run(
                mus_all=mus_all, sigmas_all=sigmas_all, num_actions=num_actions,
                n_steps=n_steps, change_every=change_every,
                pre_train_steps=damas_pre_train_steps, n_envs=damas_n_envs, verbose=False
            )
            for i in range(n_steps):
                results.append({
                    'algorithm': 'DAMAS-BO',
                    'run': run,
                    'step': i,
                    'avg_resp': float(avg_resp_damas[i]),
                    'avg_regret': float(avg_regret_damas[i]),
                    'pbest': float(pbest_damas[i]),
                    'env_id_acc': float(env_acc_damas[i])  # NaN by construction above
                })
            
        # NEW: Collect changepoint data for this run
        # True changepoints
        for cp in true_changepoints:
            changepoint_results.append({
                'run': run,
                'algorithm': 'True',
                'step': cp
            })

        # LA-MAS changepoints
        for cp in lam.changepoints:
            changepoint_results.append({
                'run': run,
                'algorithm': 'LA-MAS',
                'step': cp
            })

        # LA-MAS-BO changepoints
        for cp in lam_bo.changepoints:
            changepoint_results.append({
                'run': run,
                'algorithm': 'LA-MAS-BO',
                'step': cp
            })

        # GLR reset times
        for rt in glr.reset_times:
            changepoint_results.append({
                'run': run,
                'algorithm': 'GLR',
                'step': rt
            })

        # M-UCB reset times
        for rt in mucb.reset_times:
            changepoint_results.append({
                'run': run,
                'algorithm': 'M-UCB',
                'step': rt
            })

        print(f"Finished run {run+1}/{runs}")
    return results, changepoint_results
