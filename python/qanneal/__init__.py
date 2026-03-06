"""qanneal Python package."""

from ._qanneal import *  # noqa: F401,F403
from .solver import (  # noqa: F401
    SolveResult,
    solve,
    auto_schedule_sa,
    auto_schedule_sqa,
    auto_schedule_sa_tuned,
    auto_schedule_sqa_tuned,
    auto_ladder_sqa_tuned,
)
from .gui import launch_graph_editor  # noqa: F401

__version__ = version_string()

__all__ = [
    "__version__",
    "version_string",
    "version_major",
    "version_minor",
    "version_patch",
    "State",
    "Hamiltonian",
    "DenseIsing",
    "SparseEdge",
    "SparseIsing",
    "QUBO",
    "AnnealSchedule",
    "Observer",
    "MetricsObserver",
    "StateTraceObserver",
    "AnnealResult",
    "Annealer",
    "ReplicaResult",
    "MultiAnnealResult",
    "ReplicaAnnealer",
    "ParallelTemperingAnnealer",
    "ParallelTemperingResult",
    "SQASchedule",
    "SQAObserver",
    "SQASweepPhase",
    "SQAMetricsObserver",
    "SQAStateTraceObserver",
    "SQAResult",
    "SQAAnnealer",
    "SQAParallelTemperingAnnealer",
    "SQAParallelTemperingResult",
    "magnetization",
    "overlap",
    "SolveResult",
    "solve",
    "auto_schedule_sa",
    "auto_schedule_sqa",
    "auto_schedule_sa_tuned",
    "auto_schedule_sqa_tuned",
    "auto_ladder_sqa_tuned",
    "launch_graph_editor",
]

if "CTPIMCResult" in globals():
    __all__.append("CTPIMCResult")
if "CTPIMCAnnealer" in globals():
    __all__.append("CTPIMCAnnealer")
if "SQAParallelTemperingResult" not in globals():
    __all__.remove("SQAParallelTemperingResult")
if "SQAParallelTemperingAnnealer" not in globals():
    __all__.remove("SQAParallelTemperingAnnealer")
