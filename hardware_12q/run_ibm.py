"""
Run the 12-qubit circuit on IBM Quantum (QISKIT_IBM_TOKEN, QISKIT_IBM_INSTANCE).
Usage: python run_ibm.py ibm_kingston 300 | retrieve <job_id> <backend_name>
"""
import os
import sys

from common import circuit, save, score

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from qiskit import transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2


def service():
    return QiskitRuntimeService(channel="ibm_quantum_platform", token=os.environ["QISKIT_IBM_TOKEN"],
                                instance=os.environ.get("QISKIT_IBM_INSTANCE"))


def finish(job, name):
    res = job.result()[0]
    counts = res.data[list(res.data.keys())[0]].get_counts()
    save(name, counts, dict(job_id=job.job_id()))
    print(score(counts))


if __name__ == "__main__":
    if sys.argv[1] == "retrieve":
        finish(service().job(sys.argv[2]), sys.argv[3])
        sys.exit()
    backend_name, shots = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 300
    be = service().backend(backend_name)
    tqc = transpile(circuit(), be, optimization_level=3, seed_transpiler=7)
    print("two-qubit gates after routing:", {k: v for k, v in tqc.count_ops().items() if k in ("cz", "ecr", "cx")})
    sampler = SamplerV2(mode=be)
    sampler.options.dynamical_decoupling.enable = True
    job = sampler.run([tqc], shots=shots)
    print("job id:", job.job_id(), "(if this window closes, use: python run_ibm.py retrieve", job.job_id(), backend_name, ")")
    finish(job, backend_name)
