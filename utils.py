import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.compiler import transpile

from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

from skopt import gp_minimize
from skopt.space import Real


# =========================================================
# CONFIG
# =========================================================
MU = 2.0
SIGMA = np.sqrt(3.5)
SHOTS = 1024
N_QUBITS = 4

BO_N_CALLS = 30

PSO_ITERS = 20
PSO_NUM_PARTICLES = 16
PSO_W = 0.6
PSO_C1 = 1.4
PSO_C2 = 1.4

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# =========================================================
# NOISE MODEL
# =========================================================
def create_noise_model():

    noise_model = NoiseModel()

    error_1q = depolarizing_error(0.001, 1)
    error_2q = depolarizing_error(0.01, 2)

    noise_model.add_all_qubit_quantum_error(error_1q, ["rz", "rx"])
    noise_model.add_all_qubit_quantum_error(error_2q, ["rxx"])

    readout_error = ReadoutError([[0.98, 0.02],
                                  [0.02, 0.98]])

    noise_model.add_all_qubit_readout_error(readout_error)

    return noise_model


# =========================================================
# TARGET DISTRIBUTION
# =========================================================
def gaussian_grid(n, mu, sigma):
    xmin = mu - 4 * sigma
    xmax = mu + 4 * sigma
    return np.linspace(xmin, xmax, 2**n)


def gaussian_pdf(x, mu, sigma):
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def gaussian_target_distribution(n, probabilities):
    states = [format(i, f"0{n}b") for i in range(2**n)]
    return {state: float(p) for state, p in zip(states, probabilities)}


# =========================================================
# CIRCUIT
# =========================================================
def build_ansatz(theta, n_qubits, n_layers, pairs):

    qc = QuantumCircuit(n_qubits)

    k = 0

    for _ in range(n_layers):

        for q in range(n_qubits):

            qc.rz(theta[k], q)
            qc.rx(theta[k+1], q)
            qc.rz(theta[k+2], q)

            k += 3

        qc.barrier()

        for (i, j) in pairs:

            qc.rxx(theta[k], i, j)

            k += 1

    qc.measure_all()

    return qc


# =========================================================
# MEASUREMENT
# =========================================================
def measure_circuit(qc, mode="ideal", shots=SHOTS, seed=RANDOM_SEED):

    if mode == "ideal":

        sim = AerSimulator(seed_simulator=seed)

    elif mode == "noisy":

        noise_model = create_noise_model()

        sim = AerSimulator(
            noise_model=noise_model,
            seed_simulator=seed
        )

    else:

        raise ValueError("Unknown mode")

    qc_t = transpile(qc, sim)

    result = sim.run(qc_t, shots=shots).result()

    return result.get_counts()


def get_q(counts, n_qubits, shots=SHOTS):

    states = [format(i, f"0{n_qubits}b") for i in range(2**n_qubits)]

    q = {s: 0.0 for s in states}

    for bitstring, c in counts.items():

        q[bitstring] = c / shots

    return q


# =========================================================
# METRICS
# =========================================================
def kl_divergence(p, q, eps=1e-10):

    kl = 0.0

    for x in p.keys():

        px = p[x]
        qx = max(q.get(x, 0.0), eps)

        kl += px * np.log(px / qx)

    return float(kl)


# =========================================================
# EVALUATION
# =========================================================
def evaluate_theta(theta, p_target, n_qubits, n_layers, pairs, mode="ideal", shots=SHOTS, seed=RANDOM_SEED):

    qc = build_ansatz(theta, n_qubits=n_qubits, n_layers=n_layers, pairs=pairs)

    counts = measure_circuit(qc, mode=mode, shots=shots, seed=seed)

    q = get_q(counts, n_qubits=n_qubits, shots=shots)

    cost = kl_divergence(p_target, q)

    return cost, q


def num_parameters(n_qubits, n_layers, pairs):

    params_per_layer = 3 * n_qubits + len(pairs)

    return n_layers * params_per_layer


# =========================================================
# OPTIMIZERS
# =========================================================
def optimize_bo(p_target, n_qubits, n_layers, pairs, mode="ideal", n_calls=BO_N_CALLS, seed=RANDOM_SEED):

    total_params = num_parameters(n_qubits, n_layers, pairs)

    search_space = [Real(-np.pi, np.pi) for _ in range(total_params)]

    history = []

    def objective(theta_list):

        theta = np.array(theta_list)

        cost, _ = evaluate_theta(theta, p_target, n_qubits, n_layers, pairs, mode=mode)

        history.append(cost)

        print(f"[{mode}][BO] layer={n_layers} iter={len(history)-1} KL={cost}")

        return cost

    result = gp_minimize(objective, search_space, n_calls=n_calls, random_state=seed)

    best_theta = np.array(result.x)

    best_cost, best_q = evaluate_theta(best_theta, p_target, n_qubits, n_layers, pairs, mode=mode)

    return {"best_theta": best_theta, "best_kl": best_cost, "best_q": best_q, "history": history}


def optimize_pso(p_target, n_qubits, n_layers, pairs, mode="ideal"):

    rng = np.random.default_rng(RANDOM_SEED)

    total_params = num_parameters(n_qubits, n_layers, pairs)

    positions = rng.uniform(-np.pi, np.pi, (PSO_NUM_PARTICLES, total_params))

    velocities = rng.uniform(-0.1, 0.1, (PSO_NUM_PARTICLES, total_params))

    personal_best_positions = positions.copy()

    personal_best_scores = np.full(PSO_NUM_PARTICLES, np.inf)

    global_best_position = None

    global_best_score = np.inf

    history = []

    for it in range(PSO_ITERS):

        for i in range(PSO_NUM_PARTICLES):

            theta = positions[i]

            score, _ = evaluate_theta(theta, p_target, n_qubits, n_layers, pairs, mode=mode)

            if score < personal_best_scores[i]:

                personal_best_scores[i] = score
                personal_best_positions[i] = theta.copy()

            if score < global_best_score:

                global_best_score = score
                global_best_position = theta.copy()

        history.append(global_best_score)

        print(f"[{mode}][PSO] layer={n_layers} iter={it} best_KL={global_best_score}")

        for i in range(PSO_NUM_PARTICLES):

            r1 = rng.random(total_params)
            r2 = rng.random(total_params)

            velocities[i] = (
                PSO_W * velocities[i]
                + PSO_C1 * r1 * (personal_best_positions[i] - positions[i])
                + PSO_C2 * r2 * (global_best_position - positions[i])
            )

            positions[i] = positions[i] + velocities[i]

            positions[i] = ((positions[i] + np.pi) % (2 * np.pi)) - np.pi

    best_theta = global_best_position

    best_cost, best_q = evaluate_theta(best_theta, p_target, n_qubits, n_layers, pairs, mode=mode)

    return {"best_theta": best_theta, "best_kl": best_cost, "best_q": best_q, "history": history}


# =========================================================
# TABLE
# =========================================================
def print_results_table(results):

    print("\n" + "=" * 90)

    print(f"{'MODE':<10} {'OPTIMIZER':<10} {'ENTANGLEMENT':<12} {'LAYERS':<8} {'BEST_KL':<15}")

    print("=" * 90)

    for res in sorted(results, key=lambda r: r["best_kl"]):

        print(
            f"{res['mode']:<10}"
            f"{res['optimizer']:<10}"
            f"{res['entanglement']:<12}"
            f"{res['layers']:<8}"
            f"{res['best_kl']:<15.8f}"
        )

    print("=" * 90)


# =========================================================
# PLOTS
# =========================================================
def plot_result_distributions(results, p_target):

    ncols = 2
    nrows = int(np.ceil(len(results) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    states = list(p_target.keys())
    x = np.arange(len(states))
    width = 0.4
    px = [p_target[s] for s in states]

    for ax, res in zip(axes, results):

        q = res["best_q"]

        qx = [q.get(s, 0.0) for s in states]

        ax.bar(x - width / 2, px, width)
        ax.bar(x + width / 2, qx, width)

        ax.set_xticks(x)
        ax.set_xticklabels(states, rotation=45)

        ax.set_title(
            f"{res['mode']} | {res['optimizer']} | {res['entanglement']} | L={res['layers']}\nKL={res['best_kl']:.6f}"
        )

    plt.tight_layout()
    plt.show()


# =========================================================
# MAIN
# =========================================================
def main():

    grid = gaussian_grid(N_QUBITS, MU, SIGMA)

    pdf_values = gaussian_pdf(grid, MU, SIGMA)

    probabilities = pdf_values / np.sum(pdf_values)

    p_target = gaussian_target_distribution(N_QUBITS, probabilities)

    entanglement_configs = {

        "star": [(0, 1), (0, 2), (0, 3)],

        "circular": [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)],

    }

    layer_options = [1, 2]

    optimizer_options = ["BO", "PSO"]

    modes = ["ideal", "noisy"]

    results = []

    for mode in modes:
        for optimizer_name in optimizer_options:
            for ent_name, pairs in entanglement_configs.items():
                for n_layers in layer_options:

                    print("\n" + "-" * 80)

                    print(f"Running: mode={mode} optimizer={optimizer_name}, entanglement={ent_name}, layers={n_layers}")

                    print("-" * 80)

                    if optimizer_name == "BO":

                        opt_result = optimize_bo(p_target, N_QUBITS, n_layers, pairs, mode=mode)

                    else:

                        opt_result = optimize_pso(p_target, N_QUBITS, n_layers, pairs, mode=mode)

                    results.append({

                        "mode": mode,

                        "optimizer": optimizer_name,

                        "entanglement": ent_name,

                        "layers": n_layers,

                        "best_theta": opt_result["best_theta"],

                        "best_kl": opt_result["best_kl"],

                        "best_q": opt_result["best_q"],

                        "history": opt_result["history"],

                    })

    print_results_table(results)

    plot_result_distributions(results, p_target)


if __name__ == "__main__":
    main()