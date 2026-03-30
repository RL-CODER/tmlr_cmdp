import os
import json
import scipy
import cvxopt
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
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

class FAGK:
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
        self.theta = 1.0 / (2 * np.sqrt(self.T))
        self.zeta = 0.0
        
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
    
    def create_omega(self):
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
                        if trans_key in self.cmdp.transitions and str(s_prime) in self.cmdp.transitions[trans_key]:
                            prob = self.cmdp.transitions[trans_key][str(s_prime)]
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
                        if trans_key in self.cmdp.transitions and str(s_prime) in self.cmdp.transitions[trans_key]:
                            prob = self.cmdp.transitions[trans_key][str(s_prime)]
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
    
    def validate_omega(self, verbose=True):
        print("\n" + "="*80)
        print("Validating Omega")
        print("="*80)
        
        print(f"\n1. Basic Information:")
        print('-'*60)
        print(f"--Dimension (number of state-action pairs): {self.dim}")
        print(f"--Number of equality constraints: {self.A_eq.shape[0]}")
        print(f"--Number of inequality constraints (non-negativity): {self.dim}")
        print('-'*60)
        
        print(f"\n2. State-Action Pairs (Total: {self.dim}):")
        print('-'*60)
        for (s, a, h), idx in sorted(self.index_map.items(), key=lambda x: x[1]):
            print(f"[{idx}] rho({s},{a},{h})")
        print('-'*60)
        
        print(f"\n3. Constraints (Total: {len(self.constraint_descriptions)}):")
        print('-'*60)
        for desc in self.constraint_descriptions:
            print(f"{desc}")
        print('-'*60)
        
        print(f"\n4. Constraint Matrix Properties:")
        print('-'*60)
        print(f"--A_eq shape: {self.A_eq.shape}")
        print(f"--Rank of A_eq: {np.linalg.matrix_rank(self.A_eq)}")
        print(f"--Expected rank (based on flow conservation): {self.A_eq.shape[0] - 1}")
        print('-'*60)
        
        if verbose:
            print(f"\n5. Constraint Matrix (A_eq):")
            print('-'*60)
            for i in range(min(10, self.A_eq.shape[0])):
                nonzeros = np.where(self.A_eq[i] != 0)[0]
                if len(nonzeros) > 0:
                    print(f"Row {i}: ", end="")
                    for idx in nonzeros:
                        (s, a, h) = self.state_action_pairs[idx]
                        coeff = self.A_eq[i, idx]
                        if coeff > 0:
                            print(f"+{coeff:.2f} * rho({s},{a},{h}) ", end="")
                        else:
                            print(f"{coeff:.2f} * rho({s},{a},{h}) ", end="")
                    print(f"= {self.b_eq[i]}")
        print('-'*60)
        
        print(f"\n6. Testing Feasibility with Uniform Distribution:")
        print('-'*60)
        uniform_rho = np.ones(self.dim) / self.dim
        residuals = self.A_eq @ uniform_rho - self.b_eq
        max_residual = np.max(np.abs(residuals))
        print(f"--Max constraint violation (uniform dist): {max_residual:.6f}")
        if max_residual < 1e-6:
            print(f"Uniform distribution approximately satisfies constraints")
        else:
            print(f"Uniform distribution violates constraints by {max_residual}")
        print('-'*60)
        
        print(f"\n7. Non-negativity Constraints:")
        print('-'*60)
        min_val = np.min(uniform_rho)
        if min_val >= 0:
            print(f"All values non-negative (min: {min_val:.6f})")
        else:
            print(f"Negative values found (min: {min_val:.6f})")
        print('-'*60)
        
        print(f"\n8. Testing Projection Capability:")
        print('-'*60)
        try:
            random_point = np.random.rand(self.dim)
            
            projected = self.project_onto_omega(random_point)
            
            proj_residuals = self.A_eq @ projected - self.b_eq
            max_proj_residual = np.max(np.abs(proj_residuals))
            min_proj_val = np.min(projected)
            
            print(f"--Random point projection test:")
            print(f"*** Max constraint violation after projection: {max_proj_residual:.6f} ***")
            print(f"*** Minimum value in projected point: {min_proj_val:.6f} ***")
            
            if max_proj_residual < 1e-6 and min_proj_val >= -1e-6:
                print(f"..........Projection maintains feasibility")
            else:
                print(f"..........Projection does not maintain feasibility")
                
        except Exception as e:
            print(f"...Projection test failed: {e}")
        print('-'*60)
        
        return True
    
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
    
    def compute_optimal_rho_relaxed(self, loss_vectors, constraint_vectors):
        if len(loss_vectors) == 0 or len(constraint_vectors) == 0:
            return np.zeros(self.dim)
      
        sum_loss = np.sum(loss_vectors, axis=0)
        c = matrix(sum_loss)
        G = matrix(-np.eye(self.dim))
        h = matrix(np.zeros(self.dim))
        
        solution = solvers.lp(c, G, h, self.A, self.b)
        
        if solution['status'] == 'optimal':
            rho_opt = np.array(solution['x']).flatten()
            return rho_opt
        else:
            print(f"Optimal ρ computation failed with status: {solution['status']}")
            return np.zeros(self.dim)
    
    def run(self):
        print("\n" + "="*80)
        print("Running FAG-K")
        print("="*80)
        
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
            
            phi_prime = self.phi_prime(self.zeta)
            
            if nu_tilde > 0:
                gradient = self.omega * ell_t + phi_prime * self.omega * c_t
            else:
                gradient = self.omega * ell_t
            
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
        
        print("\nComputing optimal policy in hindsight...")
        if len(self.ell_vectors) > 0:
            loss_matrix = np.stack(self.ell_vectors)
            constraint_matrix = np.stack(self.c_vectors)
            rho_opt = self.compute_optimal_rho_relaxed(loss_matrix, constraint_matrix)
            
            optimal_cum_loss = 0.0
            for t in range(self.T):
                optimal_cum_loss += self.compute_inner_product(rho_opt, self.ell_vectors[t])
            
            final_regret = total_loss - optimal_cum_loss
            final_violation = total_violation
            
            print(f"\nAlgorithm completed!")
            
            actual_regret = []
            running_loss = 0.0
            for t in range(self.T):
                running_loss += self.learner_losses[t]
                optimal_loss_up_to_t = optimal_cum_loss * (t + 1) / self.T
                actual_regret.append(running_loss - optimal_loss_up_to_t)
        else:
            print("No data collected, returning zero regret and violation")
            actual_regret = [0.0] * self.T
            final_violation = 0.0
        
        return actual_regret, self.cum_violation

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

def compute_theoretical_bounds(t_vals, SHA):
    regret_bound = 2 * SHA * (np.sqrt(t_vals) + 1)
    violation_bound = 4 * SHA * np.sqrt(t_vals) * np.log((2 * np.sqrt(t_vals)) + 2 + (2 * t_vals))
    
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
    
    save_name = f"FAG-K_S={S}_A={A}_H={H}_T={T}.png"
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
        
        fag_k = FAGK(cmdp, T, m, adv_data, delta)
        
        fag_k.create_omega()
        fag_k.validate_omega(verbose=False)
        
        cum_regret, cum_viol = fag_k.run()
        
        regrets.append(cum_regret)
        viols.append(cum_viol)
    
    if regrets and viols:
        regrets = np.stack(regrets)
        viols = np.stack(viols)
        
        regret_mean, regret_lower, regret_upper, _ = compute_confidence_interval(regrets)
        viol_mean, viol_lower, viol_upper, _ = compute_confidence_interval(viols)
        
        t_vals = np.arange(1, T + 1)
        SHA = S * H * A
        theo_regret, theo_viol = compute_theoretical_bounds(t_vals, SHA)
        
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plt.plot(t_vals, regret_mean, color='blue', label='FAG-K Empirical Regret',
                 linestyle='dashed', linewidth=3.0)
        if num_seeds > 1:
            plt.fill_between(t_vals, regret_lower, regret_upper, alpha=0.3, color='blue')
        plt.plot(t_vals, theo_regret, color='red', label='FAG-K Theoretical Regret',
                 linestyle='solid', linewidth=3.0)
        plt.xlabel('Number of Episodes (t)', fontsize=12)
        plt.ylabel('Cumulative Regret $\mathcal{R}_{t}$', fontsize=12)
        plt.title(f'FAG-K Regret Plot: S={S}, A={A}, H={H}', fontsize=15)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        
        plt.subplot(1, 2, 2)
        plt.plot(t_vals, viol_mean, color='blue', label='FAG-K Empirical Violation',
                 linestyle='dashed', linewidth=3.0)
        if num_seeds > 1:
            plt.fill_between(t_vals, viol_lower, viol_upper, alpha=0.3, color='blue')
        plt.plot(t_vals, theo_viol, color='red', label='FAG-K Theoretical Violation',
                 linestyle='solid', linewidth=3.0)
        plt.xlabel('Number of Episodes (t)', fontsize=12)
        plt.ylabel('Cumulative Violation $\mathcal{Z}_{t}$', fontsize=12)
        plt.title(f'FAG-K Violation Plot: S={S}, A={A}, H={H}', fontsize=15)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_name, dpi=500, bbox_inches='tight')
        
        print(f"\nExperiment completed! Results saved to {save_name}")
    else:
        print("No results to plot")

if __name__ == "__main__":
    main()