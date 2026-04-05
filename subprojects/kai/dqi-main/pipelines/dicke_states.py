"""
This module contains code that has been directly extracted from the public repository:

    https://github.com/BankNatchapol/DQI-Circuit

The implementation is based on concepts presented in the paper:
    (https://arxiv.org/pdf/2504.18334)

All functions and classes included in this file are taken from the above
repository in order to support the functionality of this project.

Licensing:
----------
The original repository is distributed under the MIT License. As such, all code
within this file is also governed by the terms and conditions of the MIT License.

Acknowledgments:
----------------
Full credit for the development of this code goes to the original authors of the
DQI-Circuit repository. This file serves only as an integration point for reuse
within the present project.
"""


import math
from typing import Sequence, Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.circuit.library import RYGate
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh


class SCS2Gate(Gate):
    """
    Implements the SCS2 gate for a 2-qubit system.

    The gate is defined by the following steps:
      1. Apply a CNOT with qubit 1 as control and qubit 0 as target.
      2. Apply a controlled RY rotation on qubits 0 and 1 with rotation angle
         theta = 2 * arccos(1/sqrt(n)), where n is provided at construction.
      3. Apply a second CNOT with qubit 1 as control and qubit 0 as target.
    """

    def __init__(self, n: int) -> None:
        self.n: int = n
        super().__init__("SCS2", 2, [])

    def _define(self) -> None:
        qc = QuantumCircuit(2, name="SCS2")
        qc.cx(1, 0)
        theta: float = 2 * np.arccos(1 / np.sqrt(self.n))
        cry = RYGate(theta).control(ctrl_state="1")
        qc.append(cry, [0, 1])
        qc.cx(1, 0)
        self.definition = qc


class SCS3Gate(Gate):
    """
    Implements the SCS3 gate for a 3-qubit system.

    The gate is defined by:
      1. Applying a CNOT from qubit 2 (control) to qubit 0 (target).
      2. Applying a double-controlled RY rotation on qubits 0, 1, and 2 with
         rotation angle theta = 2 * arccos(sqrt(l/n)), where l and n are provided.
      3. Applying another CNOT from qubit 2 to qubit 0.
    """

    def __init__(self, ell: int, n: int) -> None:
        self.ell: int = ell
        self.n: int = n
        super().__init__("SCS3", 3, [])

    def _define(self) -> None:
        qc = QuantumCircuit(3, name="SCS3")
        qc.cx(2, 0)
        theta: float = 2 * np.arccos(np.sqrt(self.ell / self.n))
        ccry = RYGate(theta).control(num_ctrl_qubits=2, ctrl_state="11")
        qc.append(ccry, [0, 1, 2])
        qc.cx(2, 0)
        self.definition = qc


class SCSnkGate(Gate):
    """
    Constructs a composite SCS gate acting on k+1 qubits.

    The gate is built by composing:
      - An SCS2Gate on the first two qubits.
      - For each l in 2 to k, an SCS3Gate on qubits [0, l-1, l].

    Attributes:
      n (int): Parameter used in the rotation angles.
      k (int): Determines the number of qubits (gate acts on k+1 qubits).
    """

    def __init__(self, n: int, k: int) -> None:
        self.n: int = n
        self.k: int = k
        super().__init__(f"SCS{n}, {k}", self.k + 1, [])

    def _define(self) -> None:
        qc = QuantumCircuit(self.k + 1, name=f"SCS{self.n}, {self.k}")
        qc.append(SCS2Gate(n=self.n), [0, 1])
        for ell in range(2, self.k + 1):
            qc.append(SCS3Gate(ell=ell, n=self.n), [0, ell - 1, ell])
        self.definition = qc


class UnkStatePreparation(QuantumCircuit):
    """
    Constructs a state-preparation circuit (U) for unknown state preparation.

    The circuit is built by composing several SCSnkGates in two stages.
    The first stage applies gates on overlapping qubit subsets in descending order,
    and the second stage applies additional gates to refine the state.

    Attributes:
      n (int): Number of qubits in the circuit.
      k (int): Parameter that affects the gate construction.
    """

    def __init__(self, n: int, k: int) -> None:
        self.n: int = n
        self.k: int = k
        qc: QuantumCircuit = self.circuit()
        super().__init__(*qc.qregs, name=qc.name)
        self.compose(qc.to_gate(), qubits=self.qubits, inplace=True)

    def circuit(self) -> QuantumCircuit:
        n: int = self.n
        k: int = self.k
        label: str = f"{n}, {k}"
        qc: QuantumCircuit = QuantumCircuit(n, name=f"$U_{{{label}}}$")
        # First stage: Apply SCSnkGates on overlapping subsets (descending order)
        for ell in range(n, k, -1):
            qc.append(SCSnkGate(n=ell, k=k), list(range(n - ell, n - ell + k + 1)))
        # Second stage: Refine the state with additional SCSnkGates
        for ell in range(k, 1, -1):
            qc.append(SCSnkGate(n=ell, k=ell - 1), list(range(n - ell, n)))
        return qc


class DickeStatePreparation(QuantumCircuit):
    """
    Constructs a Dicke state preparation circuit (D) for a system with n qubits and k excitations.

    The circuit starts by initializing the first k qubits in the |1> state and then applies a series
    of SCSnkGates to entangle the qubits into a Dicke state.

    Attributes:
      n (int): Number of qubits.
      k (int): Number of excitations (qubits initially set to |1>).
    """

    def __init__(self, n: int, k: int) -> None:
        self.n: int = n
        self.k: int = k
        qc: QuantumCircuit = self.circuit()
        super().__init__(*qc.qregs, name=qc.name)
        self.compose(qc.to_gate(), qubits=self.qubits, inplace=True)

    def circuit(self) -> QuantumCircuit:
        n: int = self.n
        k: int = self.k
        label: str = f"{n}, {k}"
        qc: QuantumCircuit = QuantumCircuit(n, name=f"$D_{{{label}}}$")
        # Initialize the first k qubits to |1>
        for i in range(k):
            qc.x(i)
        # First stage: Apply SCSnkGates on overlapping subsets (descending order)
        for ell in range(n, k, -1):
            qc.append(SCSnkGate(n=ell, k=k), list(range(n - ell, n - ell + k + 1)))
        # Second stage: Refine the state with additional SCSnkGates
        for ell in range(k, 1, -1):
            qc.append(SCSnkGate(n=ell, k=ell - 1), list(range(n - ell, n)))
        return qc


class WeightedUnaryEncoding(Gate):
    """
    Weighted Unary Encoding gate for state preparation.

    This gate prepares a weighted unary state by applying a series of rotations.
    For each weight index l, the rotation angle is computed as:

        beta_l = 2 * arccos( min( sqrt(weights[l]^2 / (1 - sum_{j<l} weights[j]^2)), 1) )

    The first qubit is rotated with an RY gate and subsequent qubits are rotated
    with controlled RY (CRY) gates.

    Args:
        num_bit (int): Number of qubits.
        weights (Union[np.ndarray, Sequence[float]]): A normalized list of weights.
    """

    def __init__(
        self,
        num_bit: int,
        weights: Union[np.ndarray, Sequence[float]],
    ) -> None:
        super().__init__("Weighted_Unary", num_bit, [])
        self.num_bit: int = num_bit
        # Ensure weights are stored as a numpy array of floats.
        self.weights: np.ndarray = np.array(weights, dtype=float)

    def _define(self) -> None:
        """
        Define the internal structure of the Weighted Unary Encoding gate.
        """
        qc = QuantumCircuit(self.num_bit, name="Weighted_Unary")

        # Compute rotation angles for each weight.
        betas = []
        for ell in range(len(self.weights)):
            # Denom = 1 - sum_{j=0}^{l-1} (weights[j]^2)
            denom = 1 - np.sum(self.weights[:ell] ** 2)
            # To avoid division by zero, set ratio to 0 when denom is zero.
            ratio = (self.weights[ell] ** 2) / denom if denom > 0 else 0.0
            # Ensure the argument for arccos is at most 1.
            value = min(np.sqrt(ratio), 1.0)
            beta = 2 * np.arccos(value)
            betas.append(beta)
        betas = np.array(betas)

        # Apply the first rotation on qubit 0.
        qc.ry(betas[0], 0)
        # Apply controlled RY rotations on subsequent qubits.
        for i in range(1, self.num_bit):
            if i >= len(betas):
                continue
            if np.isnan(betas[i]) or betas[i] == 0.0:
                continue
            qc.cry(betas[i], i - 1, i)

        self.definition = qc


def get_optimal_w(m, ell, p=2, r=1):
    """
    Computes the principal eigenvector of a specific sparse tridiagonal matrix and pads it to the next power of two.

    Constructs a tridiagonal matrix A based on parameters m, l, p, and r as described in
    https://arxiv.org/pdf/2504.18334. The matrix A has a diagonal and off-diagonal structure
    reflecting interactions in a statistical or quantum model. The function extracts the
    principal (largest) eigenvector, normalizes it, and zero-pads the result to the nearest
    power-of-two length.

    Args:
        m (int): Maximum index in the state model.
        l (int): Rank or level parameter controlling the matrix size.
        p (int, optional): Parameter controlling diagonal scaling. Default is 2.
        r (int, optional): Parameter affecting normalization. Default is 1.

    Returns:
        np.ndarray: Normalized principal eigenvector, zero-padded to the next power of 2.
    """

    # build the tridiagonal entries
    d = (p - 2 * r) / np.sqrt(r * (p - r))
    diag = np.arange(ell + 1) * d
    off = np.sqrt(np.arange(1, ell + 1) * (m - np.arange(1, ell + 1) + 1))

    # <-- use a tuple here, not a list -->
    A = diags([off, diag, off], offsets=(-1, 0, 1), format="csr")

    # get principal eigenvector
    _, vecs = eigsh(A, k=1, which="LA")
    w = vecs.flatten()
    w /= np.linalg.norm(w)

    # pad up to next power of two
    orig = w.size
    target = 1 << math.ceil(math.log2(orig))
    return np.pad(w, (0, target - orig))
