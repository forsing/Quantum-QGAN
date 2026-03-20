"""
QGAN - Quantum Generative Adversarial Network
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
import random
from scipy.optimize import minimize as scipy_minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.utils import algorithm_globals

SEED = 39
np.random.seed(SEED)
random.seed(SEED)
algorithm_globals.random_seed = SEED

CSV_DRAWN = "/data/loto7hh_4582_k22.csv"
CSV_ALL   = "/data/kombinacijeH_39C7.csv"

MIN_VAL = [1, 2, 3, 4, 5, 6, 7]
MAX_VAL = [33, 34, 35, 36, 37, 38, 39]
NUM_QUBITS = 5
GEN_LAYERS = 3
DISC_LAYERS = 2
GAN_ROUNDS = 30
GEN_ITER = 50
DISC_ITER = 30


def load_draws():
    df = pd.read_csv(CSV_DRAWN)
    return df.values


def build_empirical(draws, pos):
    n_states = 1 << NUM_QUBITS
    freq = np.zeros(n_states)
    for row in draws:
        v = int(row[pos]) - MIN_VAL[pos]
        if v >= n_states:
            v = v % n_states
        freq[v] += 1
    return freq / freq.sum()


def generator_circuit(theta, n_qubits, n_layers):
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.h(i)

    idx = 0
    for layer in range(n_layers):
        for i in range(n_qubits):
            qc.ry(theta[idx], i)
            idx += 1
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        qc.cx(n_qubits - 1, 0)
        for i in range(n_qubits):
            qc.rz(theta[idx], i)
            idx += 1

    return qc


def generator_dist(theta):
    qc = generator_circuit(theta, NUM_QUBITS, GEN_LAYERS)
    sv = Statevector.from_instruction(qc)
    return sv.probabilities()


def num_gen_params():
    return GEN_LAYERS * NUM_QUBITS * 2


def discriminator_score(x, phi, n_qubits, n_layers):
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.ry(x[i] * np.pi, i)

    idx = 0
    for layer in range(n_layers):
        for i in range(n_qubits):
            qc.ry(phi[idx], i)
            idx += 1
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        for i in range(n_qubits):
            qc.rz(phi[idx], i)
            idx += 1

    sv = Statevector.from_instruction(qc)
    probs = sv.probabilities()
    return probs[0]


def num_disc_params():
    return DISC_LAYERS * NUM_QUBITS * 2


def state_to_features(state_idx, n_qubits):
    return np.array([(state_idx >> i) & 1 for i in range(n_qubits)],
                    dtype=float)


def train_discriminator(phi0, real_dist, gen_dist):
    n_states = 1 << NUM_QUBITS

    def disc_loss(phi):
        loss = 0.0
        for s in range(n_states):
            x = state_to_features(s, NUM_QUBITS)
            d_s = discriminator_score(x, phi, NUM_QUBITS, DISC_LAYERS)
            p_real = real_dist[s]
            p_gen = gen_dist[s]
            loss -= p_real * np.log(max(d_s, 1e-10))
            loss -= p_gen * np.log(max(1 - d_s, 1e-10))
        return float(loss)

    res = scipy_minimize(disc_loss, phi0, method='COBYLA',
                         options={'maxiter': DISC_ITER, 'rhobeg': 0.3})
    return res.x


def train_generator(theta0, phi):
    n_states = 1 << NUM_QUBITS

    def gen_loss(theta):
        g_dist = generator_dist(theta)
        loss = 0.0
        for s in range(n_states):
            x = state_to_features(s, NUM_QUBITS)
            d_s = discriminator_score(x, phi, NUM_QUBITS, DISC_LAYERS)
            loss += g_dist[s] * np.log(max(1 - d_s, 1e-10))
        return float(loss)

    res = scipy_minimize(gen_loss, theta0, method='COBYLA',
                         options={'maxiter': GEN_ITER, 'rhobeg': 0.3})
    return res.x


def greedy_combo(dists):
    combo = []
    used = set()
    for pos in range(7):
        ranked = sorted(enumerate(dists[pos]),
                        key=lambda x: x[1], reverse=True)
        for mv, score in ranked:
            actual = int(mv) + MIN_VAL[pos]
            if actual > MAX_VAL[pos]:
                continue
            if actual in used:
                continue
            if combo and actual <= combo[-1]:
                continue
            combo.append(actual)
            used.add(actual)
            break
    return combo


def main():
    draws = load_draws()
    print(f"Ucitano izvucenih kombinacija: {len(draws)}")

    df_all_head = pd.read_csv(CSV_ALL, nrows=3)
    print(f"Graf svih kombinacija: {CSV_ALL}")
    print(f"  Primer: {df_all_head.values[0].tolist()} ... "
          f"{df_all_head.values[-1].tolist()}")

    print(f"\n--- QGAN ({NUM_QUBITS}q, G:{GEN_LAYERS}L D:{DISC_LAYERS}L, "
          f"{GAN_ROUNDS} rundi) ---")
    print(f"  Generator params: {num_gen_params()}, "
          f"Discriminator params: {num_disc_params()}")

    dists = []
    for pos in range(7):
        print(f"  Poz {pos+1}...", end=" ", flush=True)
        real = build_empirical(draws, pos)

        theta = np.random.uniform(0, 2 * np.pi, num_gen_params())
        phi = np.random.uniform(0, 2 * np.pi, num_disc_params())

        for rnd in range(GAN_ROUNDS):
            g_dist = generator_dist(theta)
            phi = train_discriminator(phi, real, g_dist)
            theta = train_generator(theta, phi)

        final_dist = generator_dist(theta)
        dists.append(final_dist)

        top_idx = np.argsort(final_dist)[::-1][:3]
        info = " | ".join(
            f"{i + MIN_VAL[pos]}:{final_dist[i]:.3f}" for i in top_idx)
        print(f"top: {info}")

    combo = greedy_combo(dists)

    print(f"\n{'='*50}")
    print(f"Predikcija (QGAN, deterministicki, seed={SEED}):")
    print(combo)
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

"""
Ucitano izvucenih kombinacija: 4582
Graf svih kombinacija: /data/kombinacijeH_39C7.csv
  Primer: [1, 2, 3, 4, 5, 6, 7] ... [1, 2, 3, 4, 5, 6, 9]

--- QGAN (5q, G:3L D:2L, 30 rundi) ---
  Generator params: 30, Discriminator params: 20
  Poz 1... top: 2:0.526 | 1:0.469 | 3:0.001
  Poz 2... top: 9:0.188 | 11:0.157 | 8:0.153
  Poz 3... top: 15:0.156 | 14:0.155 | 11:0.155
  Poz 4... top: 26:0.229 | 25:0.165 | 24:0.143
  Poz 5... top: 27:0.437 | 24:0.310 | 21:0.142
  Poz 6... top: 33:0.544 | 34:0.454 | 26:0.000
  Poz 7... top: 37:0.562 | 38:0.430 | 7:0.006

==================================================
Predikcija (QGAN, deterministicki, seed=39):
[2, 9, x, y, z, 33, 37]
==================================================
"""



"""
QGAN - Quantum Generative Adversarial Network

Generator: 5q kolo (H + 3 sloja Ry+CX+Rz) - generise distribuciju iz superpozicije
Discriminator: 5q kolo (Ry enkodiranje + 2 sloja) - razlikuje prave od generisanih podataka
Adversarijalno treniranje: 30 rundi naizmenicnog treniranja G i D (50+30 COBYLA iter po rundi)
D uci da razlikuje empirijsku od generisane distribucije
G uci da prevari D da prihvati generisanu kao pravu
Na kraju, generatorova distribucija aproksimira empirijsku
Najslozeniji model do sad, sporiji ali mocan
Deterministicki, Statevector
"""
