"""Execution backends: local Guppy emulator vs Quantinuum Nexus (Selene / Helios)."""

from __future__ import annotations

from collections import Counter
from typing import Any, Literal

DqiExecutionBackend = Literal["local", "nexus_selene", "nexus_helios"]


def normalize_execution(name: str) -> DqiExecutionBackend:
    """Map user-facing names to an internal backend id.

    ``selene`` is treated as **local** Selene-style Guppy emulation (this machine).
    ``nexus_selene`` / ``nexus_helios`` submit jobs via ``qnexus``.
    """
    k = name.strip().lower().replace("-", "_")
    if k in ("local", "selene"):
        return "local"
    if k == "nexus_selene":
        return "nexus_selene"
    if k == "nexus_helios":
        return "nexus_helios"
    raise ValueError(
        f"Unknown execution backend {name!r}. "
        "Use 'local' or 'selene' (local emulator), 'nexus_selene', or 'nexus_helios'."
    )


def shot_to_bitstring(shot: Any, n: int) -> str:
    d = shot.as_dict()
    return "".join(str(int(d[f"m{i}"])) for i in range(n))


def bitstring_counts_from_shots(shots: Any, n: int) -> dict[str, int]:
    counts: Counter[str] = Counter(shot_to_bitstring(s, n) for s in shots)
    return dict(counts.most_common())


def run_kernel_local(kernel: Any, n: int, shots: int, seed: int) -> Any:
    emu = kernel.emulator(n_qubits=n).with_shots(int(shots)).with_seed(int(seed))
    return emu.run()


def run_kernel_nexus(
    kernel: Any,
    n: int,
    shots: int,
    *,
    mode: Literal["helios", "selene"],
    hugr_name: str,
    job_name: str,
    helios_system_name: str,
    timeout: float | None,
    max_cost: float | None = None,
) -> dict[str, int]:
    try:
        import qnexus
        from hugr.qsystem.result import QsysResult
        from quantinuum_schemas.models.backend_config import HeliosConfig, SeleneConfig
    except ImportError as exc:
        raise ImportError(
            "qnexus is required for Nexus execution (listed in cayman requirements.txt)."
        ) from exc

    kernel.check()
    pkg = kernel.compile()
    dqi_test_project = qnexus.projects.get_or_create(name="dqi-test")
    ref = qnexus.hugr.upload(pkg, name=hugr_name, project=dqi_test_project)
    cfg = (
        HeliosConfig(system_name=helios_system_name)
        if mode == "helios"
        else SeleneConfig()
    )
    exec_kwargs: dict[str, Any] = dict(
        programs=ref,
        n_shots=[int(shots)],
        backend_config=cfg,
        name=job_name,
        n_qubits=[int(n)],
        timeout=timeout,
        project=dqi_test_project,
    )
    if max_cost is not None:
        exec_kwargs["max_cost"] = float(max_cost)
    results = qnexus.execute(**exec_kwargs)
    raw = results[0]
    if not isinstance(raw, QsysResult):
        raise TypeError(
            f"Expected QsysResult from qnexus.execute, got {type(raw).__name__}. "
            "Try a different backend or check Nexus job configuration."
        )
    return bitstring_counts_from_shots(raw, n)
