'''
MaxCut Benchmark Program - CUDA Quantum
(C) Quantum Economic Development Consortium (QED-C) 2024.
'''

import os
import time

import numpy as np
from scipy.optimize import minimize

import cudaq
from typing import List

from _common import metrics as metrics
from _common.cudaq import execute as ex
from maxcut._common import common


# Saved circuit args for display (preserve qiskit module's API).
QC_ = None
benchmark_name = "MaxCut"

# Per-circuit expected distribution computed from cudaq.get_state. Keyed by
# f"_{num_qubits}_{secret_int}". Consumed by analyze_and_print_result via
# _get_expectation, then deleted.
_expectations = {}


############### Circuit Definition

@cudaq.kernel
def _qaoa_unmeasured(num_qubits: int,
                     edges_a: List[int], edges_b: List[int],
                     betas: List[float], gammas: List[float]):
    qubits = cudaq.qvector(num_qubits)
    for i in range(num_qubits):
        h(qubits[i])
    p = len(betas)
    for r in range(p):
        gamma = gammas[r]
        beta = betas[r]
        for e in range(len(edges_a)):
            i = edges_a[e]
            j = edges_b[e]
            cx(qubits[i], qubits[j])
            rz(-gamma, qubits[j])
            cx(qubits[i], qubits[j])
        for i in range(num_qubits):
            rx(2.0 * beta, qubits[i])


@cudaq.kernel
def _qaoa_measured(num_qubits: int,
                   edges_a: List[int], edges_b: List[int],
                   betas: List[float], gammas: List[float]):
    qubits = cudaq.qvector(num_qubits)
    for i in range(num_qubits):
        h(qubits[i])
    p = len(betas)
    for r in range(p):
        gamma = gammas[r]
        beta = betas[r]
        for e in range(len(edges_a)):
            i = edges_a[e]
            j = edges_b[e]
            cx(qubits[i], qubits[j])
            rz(-gamma, qubits[j])
            cx(qubits[i], qubits[j])
        for i in range(num_qubits):
            rx(2.0 * beta, qubits[i])
    mz(qubits)


def _edges_to_parallel_lists(edges):
    edges_a = [int(e[0]) for e in edges]
    edges_b = [int(e[1]) for e in edges]
    return edges_a, edges_b


def MaxCut(num_qubits, secret_int, edges, rounds, thetas_array,
           parameterized=False, measured=True):
    # parameterized is a no-op for cudaq: the @cudaq.kernel is already
    # parameterized over its runtime float args, so there's no symbolic-vs-bound
    # distinction equivalent to qiskit's ParameterVector path.

    # if no thetas_array passed in, create defaults
    if thetas_array is None:
        thetas_array = 2 * rounds * [1.0]

    # get number of qaoa rounds (p) from length of incoming array
    p = len(thetas_array) // 2

    # if rounds passed in is less than p, truncate array
    if rounds < p:
        p = rounds
        thetas_array = thetas_array[:2 * rounds]
    elif rounds > p:
        rounds = p

    # create parameters in the form expected by the ansatz generator
    # this is an array of betas followed by array of gammas, each of length = rounds
    betas = [float(t) for t in thetas_array[:p]]
    gammas = [float(t) for t in thetas_array[p:]]

    edges_a, edges_b = _edges_to_parallel_lists(edges)

    # pre-compute and save an array of expected measurements
    _compute_expectation_decorator(num_qubits, secret_int,
                                   edges_a, edges_b, betas, gammas)

    qc = [_qaoa_measured, [num_qubits, edges_a, edges_b, betas, gammas]]

    global QC_
    if QC_ is None or num_qubits <= 6:
        if num_qubits < 9:
            QC_ = qc

    return qc, None


def _compute_expectation_decorator(num_qubits, secret_int,
                                   edges_a, edges_b, betas, gammas):
    state = cudaq.get_state(_qaoa_unmeasured, num_qubits,
                            edges_a, edges_b, betas, gammas)
    n = 2 ** num_qubits
    counts = {}
    for i in range(n):
        amp = complex(state[i])
        prob = (amp.real * amp.real) + (amp.imag * amp.imag)
        if prob > 1e-12:
            key = format(i, f"0{num_qubits}b")
            counts[key] = prob
    _expectations[f"_{num_qubits}_{secret_int}"] = counts


def _get_expectation(num_qubits, degree, num_shots):
    key = f"_{num_qubits}_{degree}"
    if key not in _expectations:
        return None
    counts = _expectations[key]
    scaled = {k: round(v * num_shots) for k, v in counts.items()}
    del _expectations[key]
    return scaled


############### Result Data Analysis

def analyze_and_print_result(qc, result, num_qubits, num_shots, secret_int=None):
    counts = result.get_counts(qc)
    expected = _get_expectation(num_qubits, secret_int, num_shots)
    if expected is None:
        expected = counts
    fidelity = metrics.polarization_fidelity(counts, expected)
    return counts, fidelity


def _compute_cutsizes(result, nodes, edges):
    cnts = result.get_counts()
    cuts = list(cnts.keys())
    counts = list(cnts.values())
    sizes = [common.eval_cut(nodes, edges, cut, reverseStep=1) for cut in cuts]
    return cuts, counts, sizes


def _compute_sample_mean(counts, sizes):
    counts = np.array(counts)
    sizes = np.array(sizes)
    return -np.sum(counts * sizes) / np.sum(counts)


############### Benchmark Loop

def run(min_qubits=3, max_qubits=6,
        max_circuits=1, num_shots=100,
        method=1, rounds=1, degree=3, alpha=0.1, thetas_array=None,
        parameterized=False, do_fidelities=True,
        max_iter=30, score_metric='fidelity', x_metric='cumulative_exec_time',
        y_metric='num_qubits', fixed_metrics=None, num_x_bins=15,
        y_size=None, x_size=None,
        objective_func_type='approx_ratio', plot_results=True,
        save_res_to_file=False, save_final_counts=False,
        detailed_save_names=False, comfort=False,
        backend_id=None, provider_backend=None, eta=0.5,
        hub="ibm-q", group="open", project="main", exec_options=None,
        _instances=None,
        get_circuits=False,
        draw_circuits=True):

    if method not in (1, 2):
        raise NotImplementedError(
            f"MaxCut cudaq port supports method=1 and method=2 only (got method={method})."
        )

    print(f"{benchmark_name} ({method}) Benchmark Program - cudaq")

    metrics.init_metrics()

    _saved = {"result": None}

    def execution_handler(qc, result, num_qubits_arg, circuit_id, num_shots_arg):
        num_qubits_int = int(num_qubits_arg)
        counts, fidelity = analyze_and_print_result(
            qc, result, num_qubits_int, num_shots_arg, secret_int=int(circuit_id)
        )
        metrics.store_metric(num_qubits_int, circuit_id, 'fidelity', fidelity)
        _saved["result"] = result

    ex.init_execution(execution_handler)
    ex.set_execution_target(backend_id, provider_backend=provider_backend,
                            hub=hub, group=group, project=project,
                            exec_options=exec_options)

    for num_qubits in range(min_qubits, max_qubits + 1, 2):
        if degree < 0:
            degree = max(3, num_qubits + degree)

        instance_filename = os.path.join(
            os.path.dirname(__file__), "..", "_common",
            common.INSTANCE_DIR,
            f"mc_{num_qubits:03d}_{degree:03d}_000.txt",
        )
        nodes, edges = common.read_maxcut_instance(instance_filename, _instances)
        if nodes is None:
            print("  ... problem not found.")
            break

        if method == 1:
            print(f"************\nExecuting [{max_circuits}] circuits for num_qubits = {num_qubits}")
            for restart_ind in range(1, max_circuits + 1):
                thetas_local = thetas_array if thetas_array is not None else [1.0] * (2 * rounds)
                ts = time.time()
                qc, _params = MaxCut(num_qubits, restart_ind, edges, rounds,
                                     thetas_local, parameterized=parameterized,
                                     measured=True)
                metrics.store_metric(num_qubits, restart_ind, 'create_time', time.time() - ts)
                ex.submit_circuit(qc, num_qubits, restart_ind, shots=num_shots)

            ex.throttle_execution(metrics.finalize_group)

        else:  # method == 2 — COBYLA-driven QAOA loop
            print(f"************\nExecuting [{max_circuits}] restarts for num_qubits = {num_qubits}")
            for restart_ind in range(1, max_circuits + 1):
                state = {"iter": 0}
                init_thetas = thetas_array if thetas_array is not None else [1.0] * (2 * rounds)
                opt_ts = time.time()

                def expectation(thetas_array_iter,
                                _restart_ind=restart_ind, _state=state):
                    unique_id = _restart_ind * 1000 + _state["iter"]
                    metrics.store_metric(num_qubits, unique_id, 'thetas_array',
                                         list(map(float, thetas_array_iter)))

                    ts2 = time.time()
                    qc, _params = MaxCut(num_qubits, unique_id, edges, rounds,
                                         list(map(float, thetas_array_iter)),
                                         parameterized=parameterized, measured=True)
                    metrics.store_metric(num_qubits, unique_id, 'create_time',
                                         time.time() - ts2)
                    metrics.store_metric(num_qubits, unique_id, 'rounds', rounds)
                    metrics.store_metric(num_qubits, unique_id, 'degree', degree)

                    ex.submit_circuit(qc, num_qubits, unique_id, shots=num_shots)

                    result = _saved["result"]
                    cuts, counts, sizes = _compute_cutsizes(result, nodes, edges)
                    obj = _compute_sample_mean(counts, sizes)

                    _state["iter"] += 1
                    return obj

                res = minimize(expectation, init_thetas, method='COBYLA',
                               options={'maxiter': max_iter})
                final_id = restart_ind * 1000 + 0
                metrics.store_metric(num_qubits, final_id, 'opt_exec_time',
                                     time.time() - opt_ts)

            metrics.process_circuit_metrics_2_level(num_qubits)
            metrics.finalize_group(str(num_qubits))

    if method == 1:
        ex.finalize_execution(metrics.finalize_group)
    else:
        ex.finalize_execution(None, report_end=True)


def kernel_draw():
    print("Sample Circuit:")
    if QC_ is not None:
        try:
            print(cudaq.draw(QC_))
        except Exception as ex:
            print(f"ERROR attempting to draw the kernel")
            print(ex)
    else:
        print("  ... too large!")


def load_data_and_plot(folder=None, backend_id=None, **kwargs):
    print("load_data_and_plot is not implemented for the cudaq maxcut port.")
