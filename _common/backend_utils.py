def is_simulator_backend(api=None, backend_id=None, provider_backend=None):
    if provider_backend is not None:
        simulator = getattr(provider_backend, "simulator", None)
        if simulator is not None:
            return bool(simulator)
        configuration = getattr(provider_backend, "configuration", None)
        if callable(configuration):
            simulator = getattr(configuration(), "simulator", None)
            if simulator is not None:
                return bool(simulator)

    api_name = (api or "qiskit").lower()
    backend = (backend_id or "").lower()

    if api_name == "cudaq":
        return backend_id is None or backend in {
            "cudaq_simulator", "nvidia", "qpp", "qpp-cpu", "density-matrix-cpu"
        } or "simulator" in backend

    if api_name == "qiskit":
        return backend_id is None or backend in {
            "qasm_simulator", "statevector_simulator", "aer_simulator",
            "aer_sampler", "statevector_sampler"
        } or "simulator" in backend

    return backend_id is None or "simulator" in backend


def api_display_name(api=None):
    api_name = (api or "qiskit").lower()
    return {
        "cudaq": "CUDA-Q",
        "qiskit": "Qiskit",
        "cirq": "Cirq",
        "braket": "Braket",
        "ocean": "Ocean",
    }.get(api_name, api or "Qiskit")
