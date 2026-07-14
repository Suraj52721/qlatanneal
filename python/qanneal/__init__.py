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
    j_perp_from_beta_gamma,
    optimal_j_perp_params,
    j_rms_from_problem,
)


def launch_graph_editor() -> None:
    """Launch the optional Tk GUI."""
    from .gui import launch_graph_editor as _launch_graph_editor

    _launch_graph_editor()

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
    "HigherOrderIsing",
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
    "SQAChiResult",
    "SQAChiAnnealer",
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
    "j_perp_from_beta_gamma",
    "optimal_j_perp_params",
    "j_rms_from_problem",
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
