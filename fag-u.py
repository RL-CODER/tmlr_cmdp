import os
import json
import scipy
import cvxopt
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from math import log, sqrt
from cvxopt import matrix, solvers
from collections import defaultdict

solvers.options['show_progress'] = False

def create_CMDP_json(filename, S, A, H, constraint_type):
    transitions = {}
    available_states = S[:]
    np.random.shuffle(available_states)
    
    first_layer = [available_states.pop()]
    last_layer = [available_states.pop()]
    middle_layers = [[] for _ in range(H - 2)]
    
    for i in range(H - 2):
        s = available_states.pop()
        middle_layers[i].append(s)
    
    for s in available_states:
        index = np.random.randint(0, H - 2)
        middle_layers[index].append(s)
    
    all_layers = [first_layer] + middle_layers + [last_layer]
    s0 = all_layers[0][0]
    
    state_to_layer = {}
    for layer_index, layer in enumerate(all_layers):
        for state in layer:
            state_to_layer[state] = layer_index
    
    det_path_states = []
    det_path_actions = {}
    det_actions = []
    if constraint_type == "adv":
        det_path_states = [layer[0] for layer in all_layers]
        det_path_actions = {s: np.random.choice(A) for s in det_path_states[:-1]}
        det_actions = [(int(s), int(a)) for s, a in det_path_actions.items()]
    
    for k in range(len(all_layers) - 1):
        l_k = all_layers[k]
        l_k1 = all_layers[k + 1]
        for s in l_k:
            for a in A:
                if constraint_type == "adv" and s in det_path_states[:-1] and a == det_path_actions[s]:
                    chosen_next = det_path_states[k + 1]
                    probs = [1.0 if s_prime == chosen_next else 0.0 for s_prime in l_k1]
                else:
                    probs = np.random.dirichlet(np.ones(len(l_k1)))
                transitions[f"{s}_{a}"] = {str(s_prime): float(p) for s_prime, p in zip(l_k1, probs)}
                
    cmdp = {
        "S": S,
        "A": A,
        "H": H,
        "all_layers": all_layers,
        "state_to_layer": state_to_layer,
        "transitions": transitions,
        "s0": s0
    }
    
    if constraint_type == "adv":
        cmdp["det_actions"] = det_actions
    
    with open(filename, 'w') as f:
        json.dump(cmdp, f, indent=4)

class OGD:
    def __init__(self, A, eta):
        self.A = A
        self.eta = eta
        self.weights = np.ones(A) / A
    
    def predict(self):
        return self.weights
    
    def update(self, gradient_vector):
        self.weights -= self.eta * np.array(gradient_vector)
        self.weights = self._project_to_simplex(self.weights)
    
    @staticmethod
    def _project_to_simplex(v):
        v = np.array(v)
        if np.sum(v) == 1 and np.all(v >= 0):
            return v
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, len(u) + 1) > (cssv - 1))[0][-1]
        theta = (cssv[rho] - 1) / (rho + 1)
        w = np.maximum(v - theta, 0)
        
        return w

class CMDP:
    def __init__(self, json_path):
        with open(json_path) as f:
            data = json.load(f)
        
        self.H = data["H"]
        self.S = data["S"]
        self.A = data["A"]
        self.all_layers = data["all_layers"]
        self.state_to_layer = {int(k): v for k, v in data["state_to_layer"].items()}
        self.transitions = data["transitions"]
        self.s0 = data["s0"]
        self.det_actions = set(tuple(item) for item in data.get("det_actions", []))
        self.adversarial_reward_vectors = None
        self.adversarial_constraint_vectors = None
    
    def get_next_state(self, s, a):
        key = f"{s}_{a}"
        if key not in self.transitions:
            raise ValueError(f"Transition not defined for the pair: ({s}, {a}).")
        
        probs_dict = self.transitions[key]
        next_states = list(probs_dict.keys())
        probs = list(probs_dict.values())
        
        return int(np.random.choice(next_states, p=probs))

class AdversarialDataGenerator:
    def __init__(self, S, A, m, eta=0.01, adv_reward=True, adv_constraints=True):
        self.S = S
        self.A = A
        self.m = m
        self.eta = eta
        self.adv_reward = adv_reward
        self.adv_constraints = adv_constraints
        self.rng = np.random.default_rng()
        
        if self.adv_constraints:
            self.constraint_learners = [[OGD(A, eta) for _ in range(m)] for _ in range(S)]
        if self.adv_reward:
            self.reward_learners = [OGD(A, eta) for _ in range(S)]
        
        self.true_reward_vectors = {s: self.rng.uniform(0, 1, size=A) for s in range(S)}
        self.true_constraint_vectors = {s: [self.rng.uniform(-1, 1, size=A) for _ in range(m)] for s in range(S)}
    
    def get_adversarial_data(self, policy, t=None):
        reward_vectors = {}
        constraint_vectors = {}
        
        for s in range(self.S):
            pi_s = np.array([policy.get((s, a), 0.0) for a in range(self.A)])
            
            if self.adv_reward:
                grad = -self.true_reward_vectors[s] * pi_s
                self.reward_learners[s].update(grad)
                reward_vectors[s] = self.reward_learners[s].predict()
            else:
                reward_vectors[s] = self.true_reward_vectors[s]
            
            constraint_vectors[s] = []
            for i in range(self.m):
                if self.adv_constraints:
                    grad_c = -self.true_constraint_vectors[s][i] * pi_s
                    self.constraint_learners[s][i].update(grad_c)
                    constraint_vectors[s].append(self.constraint_learners[s][i].predict())
                else:
                    constraint_vectors[s].append(self.true_constraint_vectors[s][i])
        
        return reward_vectors, constraint_vectors

class TransitionConfidenceSet:
    def __init__(self, S, A, H, delta):
        self.S = S
        self.A = A
        self.H = H
        self.delta = delta
        self.X = defaultdict(int)
        self.Y = defaultdict(int)
        self.P_hat = {}
        self.P_set = []
        self.epoch = 1
        self.X_prev = defaultdict(int)
        
    def update(self, s, a, s_next):
        key = (s, a)
        self.X[key] += 1
        self.Y[(s, a, s_next)] += 1
    
    def maybe_new_epoch(self):
        new_epoch = False
        for (s, a), count in self.X.items():
            if count >= max(1, 2 * self.X_prev.get((s, a), 0)):
                new_epoch = True
                break
        
        if new_epoch:
            self.epoch += 1
            self.X_prev = self.X.copy()
            self._update_confidence_set()
        
        return new_epoch
    
    def _update_confidence_set(self):
        self.P_hat = {}
        for (s, a, s_next), count in self.Y.items():
            key = (s, a)
            if key not in self.P_hat:
                self.P_hat[key] = {}
            self.P_hat[key][s_next] = count / max(1, self.X[key])
        
        self.P_set = []
        epsilon = 2 * sqrt(log(self.S * self.A * 1000 / self.delta) / max(1, min(self.X.values()) - 1))
        self.P_set.append(self.P_hat)
    
    def get_transition_estimate(self):
        return self.P_hat

class FAGU:
    def __init__(self, cmdp, T, m, adv_data=None, delta=0.01):
        self.cmdp = cmdp
        self.S = cmdp.S
        self.A = cmdp.A
        self.num_states = len(self.S)
        self.num_actions = len(self.A)
        self.T = T
        self.m = m
        self.H = cmdp.H
        self.adv_data = adv_data
        self.delta = delta
        
        self.SHA = self.num_states * self.H * self.num_actions
        self.L = np.sqrt(self.SHA)
        self.D = np.sqrt(self.SHA)
        self.omega = 1.0 / (2 * self.L * self.D)
        self.k_t = (np.sqrt(self.T) + (np.sqrt(((self.num_states * self.H * self.T) / 4) * np.log(self.num_states * self.num_actions * self.T / self.delta))))
        self.theta = 1.0 / (2 * self.k_t)
        self.zeta = 0.0
        
        self.transition_confidence = TransitionConfidenceSet(
            self.num_states, self.num_actions, self.H, delta
        )
        
        self.gradient_norms = []
        self.episode = 0
        self.rho_vector = None
        self.ell_vectors = []
        self.c_vectors = []
        self.learner_losses = []
        self.learner_constraints = []
        self.cum_gradient_norm_squared = 0.0
        self.cum_loss = 0.0
        self.cum_constraint_violation = 0.0
        self.cum_regret = []
        self.cum_violation = []
        
        self.current_transitions = None
    
    def init_uniform_rho_vector(self):
        rho_vector = np.zeros(self.dim)
        s0 = self.cmdp.s0
        for a in range(self.num_actions):
            if (s0, a, 0) in self.index_map:
                idx = self.index_map[(s0, a, 0)]
                rho_vector[idx] = 1.0 / self.num_actions
        rho_vector = self.project_onto_omega(rho_vector)
        
        return rho_vector
    
    def extract_policy_from_rho_vector(self, rho_vector):
        policy = {}
        marginal = defaultdict(float)
        
        for (s, a, h), idx in self.index_map.items():
            if rho_vector[idx] > 0:
                marginal[s] += rho_vector[idx]
        
        for (s, a, h), idx in self.index_map.items():
            d_s = marginal.get(s, 0)
            if d_s > 1e-12:
                policy[(s, a)] = rho_vector[idx] / d_s
            else:
                policy[(s, a)] = 1.0 / self.num_actions
        
        return policy
    
    def convert_adversarial_vectors_to_dict(self, reward_vectors, constraint_vectors):
        reward_dict = defaultdict(float)
        constraint_dict = defaultdict(float)
        
        for s in range(self.num_states):
            if s in reward_vectors:
                for a in range(self.num_actions):
                    reward_dict[(s, a)] = reward_vectors[s][a]
            
            if s in constraint_vectors and len(constraint_vectors[s]) > 0:
                for a in range(self.num_actions):
                    constraint_dict[(s, a)] = constraint_vectors[s][0][a]
        
        return reward_dict, constraint_dict
    
    def get_adversarial_vectors(self, policy):
        reward_vectors, constraint_vectors = self.adv_data.get_adversarial_data(policy)
        loss_dict, constraint_dict = self.convert_adversarial_vectors_to_dict(reward_vectors, constraint_vectors)
        
        loss_vec = np.zeros(self.dim)
        constraint_vec = np.zeros(self.dim)
        
        for (s, a, h), idx in self.index_map.items():
            loss_vec[idx] = loss_dict.get((s, a), 0.0)
            constraint_vec[idx] = constraint_dict.get((s, a), 0.0)
        
        return loss_vec, constraint_vec
    
    def compute_inner_product(self, rho_vector, vector):
        return np.dot(rho_vector, vector)
    
    def create_omega_with_transitions(self, transitions):
        self.index_map = {}
        idx = 0
        self.state_action_pairs = []
        
        for h in range(self.H):
            layer_states = self.cmdp.all_layers[h]
            for s in layer_states:
                for a in range(self.num_actions):
                    self.index_map[(s, a, h)] = idx
                    self.state_action_pairs.append((s, a, h))
                    idx += 1
        self.dim = len(self.state_action_pairs)
        
        num_eq_constraints = 1
        for h in range(1, self.H):
            num_eq_constraints += len(self.cmdp.all_layers[h])
        
        self.A_eq = np.zeros((num_eq_constraints, self.dim))
        self.b_eq = np.zeros(num_eq_constraints)
        
        self.constraint_descriptions = []
        
        constraint_idx = 0
        s0 = self.cmdp.s0
        desc = f"Constraint {constraint_idx}: Initial state constraint - "
        desc += " + ".join([f"ρ({s0},{a},0)" for a in range(self.num_actions)]) + " = 1"
        self.constraint_descriptions.append(desc)
        
        for a in range(self.num_actions):
            if (s0, a, 0) in self.index_map:
                self.A_eq[constraint_idx, self.index_map[(s0, a, 0)]] = 1.0
        self.b_eq[constraint_idx] = 1.0
        
        for h in range(self.H - 1):
            layer_states_h = self.cmdp.all_layers[h]
            layer_states_h1 = self.cmdp.all_layers[h + 1]
            for s_prime in layer_states_h1:
                constraint_idx += 1
                desc = f"Constraint {constraint_idx}: Flow constraint at h={h+1}, state s'={s_prime} - "
                lhs_terms = []
                for a_prime in range(self.num_actions):
                    if (s_prime, a_prime, h + 1) in self.index_map:
                        lhs_terms.append(f"ρ({s_prime},{a_prime},{h+1})")
                desc += " + ".join(lhs_terms) + " = "
                
                rhs_terms = []
                for s in layer_states_h:
                    for a in range(self.num_actions):
                        trans_key = f"{s}_{a}"
                        if trans_key in transitions and str(s_prime) in transitions[trans_key]:
                            prob = transitions[trans_key][str(s_prime)]
                        else:
                            prob = 1.0 / len(layer_states_h1)
                        
                        if prob > 0:
                            if (s, a, h) in self.index_map:
                                rhs_terms.append(f"{prob:.3f}*ρ({s},{a},{h})")
                desc += " + ".join(rhs_terms)
                self.constraint_descriptions.append(desc)
                
                for a_prime in range(self.num_actions):
                    if (s_prime, a_prime, h + 1) in self.index_map:
                        self.A_eq[constraint_idx, self.index_map[(s_prime, a_prime, h + 1)]] = 1.0
                
                for s in layer_states_h:
                    for a in range(self.num_actions):
                        trans_key = f"{s}_{a}"
                        if trans_key in transitions and str(s_prime) in transitions[trans_key]:
                            prob = transitions[trans_key][str(s_prime)]
                        else:
                            prob = 1.0 / len(layer_states_h1)
                        
                        if (s, a, h) in self.index_map:
                            self.A_eq[constraint_idx, self.index_map[(s, a, h)]] -= prob
        
        self.A_ineq = -np.eye(self.dim)
        self.b_ineq = np.zeros(self.dim)
        
        self.P = matrix(np.eye(self.dim))
        self.G = matrix(self.A_ineq)
        self.h = matrix(self.b_ineq)
        self.A = matrix(self.A_eq)
        self.b = matrix(self.b_eq)
        
        return self.index_map, self.A_eq, self.b_eq
    
    def project_onto_omega(self, y):
        q = matrix(-y)
        solution = solvers.qp(self.P, q, self.G, self.h, self.A, self.b)
        
        if solution['status'] == 'optimal':
            x_opt = np.array(solution['x']).flatten()
            return x_opt
        else:
            print(f"Projection failed with status: {solution['status']}")
            return y
    
    def phi(self, zeta):
        return np.exp(self.theta * zeta) - 1
    
    def phi_prime(self, zeta):
        return self.theta * np.exp(self.theta * zeta)
    
    def compute_gradient(self, ell_t, c_t):
        constraint_val = self.compute_inner_product(self.rho_vector, c_t)
        
        if constraint_val > 0:
            gradient = self.omega * ell_t + self.phi_prime(self.zeta) * self.omega * c_t
        else:
            gradient = self.omega * ell_t
        
        return gradient
    
    def run(self):
        print("\n" + "="*80)
        print("Running FAG-U")
        print("="*80)
        
        initial_transitions = {}
        for h in range(self.H - 1):
            layer_states_h = self.cmdp.all_layers[h]
            layer_states_h1 = self.cmdp.all_layers[h + 1]
            for s in layer_states_h:
                for a in range(self.num_actions):
                    trans_key = f"{s}_{a}"
                    uniform_prob = 1.0 / len(layer_states_h1)
                    initial_transitions[trans_key] = {
                        str(s_prime): uniform_prob for s_prime in layer_states_h1
                    }
        
        self.create_omega_with_transitions(initial_transitions)
        
        self.rho_vector = self.init_uniform_rho_vector()
        self.zeta = 0.0
        self.cum_gradient_norm_squared = 0.0
        
        self.ell_vectors = []
        self.c_vectors = []
        self.learner_losses = []
        self.learner_constraints = []
        self.cum_regret = []
        self.cum_violation = []
        
        total_loss = 0.0
        total_violation = 0.0
        
        for t in range(1, self.T + 1):
            policy = self.extract_policy_from_rho_vector(self.rho_vector)
            trajectory = []
            s = self.cmdp.s0
            
            for h in range(self.H - 1):
                action_probs = [policy.get((s, a), 0.0) for a in range(self.num_actions)]
                a = np.random.choice(range(self.num_actions), p=action_probs)
                
                s_next = self.cmdp.get_next_state(s, a)
                
                trajectory.append((s, a, s_next))
                
                self.transition_confidence.update(s, a, s_next)
                
                s = s_next
            
            if self.transition_confidence.maybe_new_epoch():
                new_transitions = self.transition_confidence.get_transition_estimate()
                trans_dict = {}
                for (s, a), probs in new_transitions.items():
                    trans_dict[f"{s}_{a}"] = {str(s_prime): p for s_prime, p in probs.items()}
                self.create_omega_with_transitions(trans_dict)
            
            ell_t, c_t = self.get_adversarial_vectors(policy)
            self.ell_vectors.append(ell_t)
            self.c_vectors.append(c_t)
            
            mu_t = self.compute_inner_product(self.rho_vector, ell_t)
            nu_t = self.compute_inner_product(self.rho_vector, c_t)
            
            self.learner_losses.append(mu_t)
            self.learner_constraints.append(nu_t)
            
            total_loss += mu_t
            total_violation += max(0, nu_t)
            
            mu_tilde = self.omega * mu_t
            nu_tilde = self.omega * max(0, nu_t)
            
            self.zeta += nu_tilde
            
            gradient = self.compute_gradient(ell_t, c_t)
            
            grad_norm_squared = np.dot(gradient, gradient)
            self.cum_gradient_norm_squared += grad_norm_squared
            self.gradient_norms.append(np.sqrt(grad_norm_squared))
            
            if self.cum_gradient_norm_squared > 0:
                eta_t = (np.sqrt(2) * self.D) / (2 * np.sqrt(self.cum_gradient_norm_squared))
            else:
                eta_t = 1.0
            
            self.rho_vector = self.rho_vector - eta_t * gradient
            self.rho_vector = self.project_onto_omega(self.rho_vector)
            
            self.cum_regret.append(total_loss)
            self.cum_violation.append(total_violation)
            
            if t % 500 == 0 or t == self.T:
                print(f"Episode {t}/{self.T}")
        
        print(f"\nAlgorithm completed!")
        
        return self.cum_regret, self.cum_violation

def compute_confidence_interval(data, confidence=0.95):
    if len(data) == 0:
        return np.array([]), np.array([]), np.array([]), 0
    
    data_array = np.array(data)
    mean = np.mean(data_array, axis=0)
    std = np.std(data_array, axis=0, ddof=1)
    n = len(data)
    
    if n > 1:
        t_value = stats.t.ppf((1 + confidence) / 2, df=n-1)
        margin_of_error = t_value * std / np.sqrt(n)
    else:
        margin_of_error = 0
    
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error
    
    return mean, lower_bound, upper_bound, margin_of_error

def compute_theoretical_bounds(t_vals, S, A, H, SHA, delta, transition_type="unknown"):
    if transition_type == "unknown":
        regret_bound = 2 * SHA * (1 + np.sqrt(t_vals) + np.sqrt(((S * H * t_vals) / 4) * np.log(S * A * t_vals / delta)))
        violation_bound = 2 * SHA * ((2 * np.sqrt(t_vals)) + (2 * np.sqrt(((S * H * t_vals) / 4) * np.log(S * A * t_vals / delta)))) * np.log(2 + (2 * t_vals) + np.sqrt(t_vals) + np.sqrt(((S * H * t_vals) / 4) * np.log(S * A * t_vals / delta)))
    
    return regret_bound, violation_bound

def main():
    S = 4
    A = 3
    H = 4
    m = 1
    T = 50000
    delta = 0.01
    reward_type = "adv"
    constraint_type = "adv"
    
    save_name = f"FAG-U_S={S}_A={A}_H={H}_T={T}.png"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(script_dir, "cmdp.json")
    
    num_seeds = 5
    seeds = range(num_seeds)
    regrets = []
    viols = []
    
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Running experiment with seed {seed}")
        print('='*60)
        
        np.random.seed(seed)
        create_CMDP_json(filename, list(range(S)), list(range(A)), H, constraint_type)
        cmdp = CMDP(filename)
        
        adv_data = AdversarialDataGenerator(
            S, A, m,
            eta=0.01,
            adv_reward=(reward_type == "adv"),
            adv_constraints=(constraint_type == "adv")
        )
        
        fag_u = FAGU(cmdp, T, m, adv_data, delta)
        cum_regret, cum_viol = fag_u.run()
        
        regrets.append(cum_regret)
        viols.append(cum_viol)
    
    if regrets and viols:
        regrets = np.stack(regrets)
        viols = np.stack(viols)
        
        regret_mean, regret_lower, regret_upper, _ = compute_confidence_interval(regrets)
        viol_mean, viol_lower, viol_upper, _ = compute_confidence_interval(viols)
        
        t_vals = np.arange(1, T + 1)
        SHA = S * H * A
        theo_regret, theo_viol = compute_theoretical_bounds(t_vals, S, A, H, SHA, delta, "unknown")
        
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plt.plot(t_vals, regret_mean, color='blue', label='FAG-U Empirical Regret',
                 linestyle='dashed', linewidth=3.0)
        if num_seeds > 1:
            plt.fill_between(t_vals, regret_lower, regret_upper, alpha=0.3, color='blue')
        plt.plot(t_vals, theo_regret, color='red', label='FAG-U Theoretical Regret',
                 linestyle='solid', linewidth=3.0)
        plt.xlabel('Number of Episodes (t)', fontsize=12)
        plt.ylabel('Cumulative Regret $\mathcal{R}_{t}$', fontsize=12)
        plt.title(f'FAG-U Regret Plot: S={S}, A={A}, H={H}', fontsize=15)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        
        plt.subplot(1, 2, 2)
        plt.plot(t_vals, viol_mean, color='blue', label='FAG-U Empirical Violation',
                 linestyle='dashed', linewidth=3.0)
        if num_seeds > 1:
            plt.fill_between(t_vals, viol_lower, viol_upper, alpha=0.3, color='blue')
        plt.plot(t_vals, theo_viol, color='red', label='FAG-U Theoretical Violation',
                 linestyle='solid', linewidth=3.0)
        plt.xlabel('Number of Episodes (t)', fontsize=12)
        plt.ylabel('Cumulative Violation $\mathcal{Z}_{t}$', fontsize=12)
        plt.title(f'FAG-U Violation Plot: S={S}, A={A}, H={H}', fontsize=15)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_name, dpi=500, bbox_inches='tight')
        
        print(f"\nExperiment completed! Results saved to {save_name}")
    else:
        print("No results to plot")

if __name__ == "__main__":
    main()