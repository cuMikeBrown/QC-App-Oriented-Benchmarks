'''
Amplitude Estimation Benchmark Program - CUDA Quantum Kernel
(C) Quantum Economic Development Consortium (QED-C) 2024.
'''

import math
import cudaq

# saved circuit for display
QC_ = None

############### Circuit Definition

# Construct A operator that takes |0>_{n+1} to sqrt(1-a) |psi_0>|0> + sqrt(a) |psi_1>|1>
@cudaq.kernel
def A_op(state_obj: cudaq.qview, num_state_qubits: int, theta: float):
    ry(theta, state_obj[0])
    for i in range(num_state_qubits):
        x.ctrl(state_obj[0], state_obj[i + 1])


@cudaq.kernel
def A_op_inv(state_obj: cudaq.qview, num_state_qubits: int, theta: float):
    for i in range(num_state_qubits):
        x.ctrl(state_obj[0], state_obj[i + 1])
    ry(-theta, state_obj[0])


# Construct the grover-like operator (-S_chi, A_inv, S_0, A)
@cudaq.kernel
def Q_op(state_obj: cudaq.qview, num_state_qubits: int, theta: float):
    # -S_chi
    x(state_obj[0])
    z(state_obj[0])
    x(state_obj[0])

    # A_circ_inverse
    A_op_inv(state_obj, num_state_qubits, theta)

    # S_0
    for i in range(num_state_qubits + 1):
        x(state_obj[i])
    h(state_obj[0])
    if num_state_qubits == 1:
        x.ctrl(state_obj[1], state_obj[0])
    else:
        x.ctrl(state_obj[1:num_state_qubits + 1], state_obj[0])
    h(state_obj[0])
    for i in range(num_state_qubits + 1):
        x(state_obj[i])

    # A_circ
    A_op(state_obj, num_state_qubits, theta)


@cudaq.kernel
def ae_iqft(register: cudaq.qview):
    M_PI = 3.1415926536
    input_size = register.size()
    for i_qubit in range(input_size):
        ri_qubit = input_size - i_qubit - 1
        h(register[ri_qubit])
        num_crzs = input_size - i_qubit - 1
        if i_qubit < input_size - 1:
            for j in range(num_crzs):
                divisor = 2 ** (j + 1)
                r1.ctrl(-M_PI / divisor,
                        register[ri_qubit], register[ri_qubit - j - 1])


@cudaq.kernel
def ae_kernel(num_state_qubits: int, num_counting_qubits: int, theta: float):
    num_qubits = num_state_qubits + 1 + num_counting_qubits
    qubits = cudaq.qvector(num_qubits)

    # Prepare state from A, and counting qubits with H transform
    A_op(qubits[0:num_state_qubits + 1], num_state_qubits, theta)
    for i in range(num_state_qubits + 1, num_qubits):
        h(qubits[i])

    repeat = 1
    for j in range(num_counting_qubits):
        for _ in range(repeat):
            cudaq.control(Q_op,
                          qubits[num_state_qubits + 1 + j],
                          qubits[0:num_state_qubits + 1],
                          num_state_qubits, theta)
        repeat = repeat * 2

    # inverse quantum Fourier transform only on counting qubits
    ae_iqft(qubits[num_state_qubits + 1:num_qubits])

    # measure counting qubits
    mz(qubits[num_state_qubits + 1:num_qubits])


def AmplitudeEstimation(num_state_qubits: int, num_counting_qubits: int, a,
                        psi_zero=None, psi_one=None):
    theta = 2.0 * math.asin(math.sqrt(a))
    qc = [ae_kernel, [num_state_qubits, num_counting_qubits, theta], {"counts_dict": True}]

    global QC_
    if num_counting_qubits + num_state_qubits + 1 <= 6:
        QC_ = qc

    return qc

############### Circuit Drawer

# Draw the circuits of this benchmark program
def kernel_draw():
    print("Sample Circuit:")
    if QC_ is not None:
        try:
            print(cudaq.draw(QC_[0], *QC_[1]))
        except Exception as ex:
            print(f"ERROR attempting to draw the kernel")
            print(ex)
    else:
        print("  ... too large!")
