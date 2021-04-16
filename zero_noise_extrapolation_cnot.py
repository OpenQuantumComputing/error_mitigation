from qiskit import QuantumCircuit, execute, Aer, transpile

from qiskit.result.result import Result
from qiskit.providers.aer.noise import NoiseModel

from qiskit.transpiler import PassManager, PassManagerConfig, CouplingMap
from qiskit.transpiler.passes import Unroller, Optimize1qGates
from qiskit.transpiler.preset_passmanagers.level3 import level_3_pass_manager

from numpy import asarray, ndarray, shape, zeros, empty, average, transpose, dot, sqrt
from numpy.linalg import solve

import random, os, pickle, sys, errno
from dataclasses import dataclass

abs_path = os.path.dirname(__file__)
sys.path.append(abs_path)
sys.path.append(os.path.dirname(abs_path))

from typing import Callable, Union

"""
-- ZERO NOISE EXTRAPOLATION for CNOT-gates --

This is an implementation of the zero-noise extrapolation technique for quantum error mitigation. The goal is to
mitigate noise present in a quantum device when evaluating some expectation value that is computed by a quantum circuit
with subsequent measurements. The main idea of zero-noise extrapolation is to amplify the noise by a set of known
noise amplification factors, such as to obtain a set of noise amplified expectation values. Richardsson extrapolation
is then used to extrapolate the expectation value to the zero-noise limit.

The noise that is here amplified and mitigated is specifically general noise in CNOT-gates. Note that in modern quantum
devices the noise in the multi-qubit CNOT-gates tend to be an order of magnitude larger than in single-qubit gates.

To amplify the noise we use that the CNOT-gate is its own inverse, i.e., CNOT*CNOT = Id, where Id is the identity gate.
Thus an odd number of CNOT-gates in a series will in the noise-less case have the same action as a single CNOT-gate.
The noise is amplified by replacing each CNOT-gate in the original bare circuit with a series of (2*i + 1) CNOT's,
using noise amplification factors c=1, 3, 5, ..., 2*n - 1, for n being the total number of amplification factors.
For c=3, each CNOT is thus replaced by the sequence CNOT*CNOT*CNOT, and while this has the same action in the noise-less
case, in the noisy case the noise operation associated with the noisy CNOT-gate will be applied thrice instead of once.

"""


# Dataclasses, containing partial (noise amplified) and final results

@dataclass(frozen=True)
class NoiseAmplifiedResult:
    amp_factor: int
    shots: int
    qc: QuantumCircuit
    depth: int
    exp_val: float
    variance: float


@dataclass(frozen=True)
class ZeroNoiseExtrapolationResult:
    qc: QuantumCircuit
    noise_amplified_results: ndarray
    noise_amplification_factors: ndarray
    gamma_coefficients: ndarray
    exp_val: float

    @property
    def bare_exp_val(self) -> float:
        return self.noise_amplified_results[0].exp_val

    @property
    def noise_amplified_exp_vals(self) -> ndarray:
        return asarray([result.exp_val for result in self.noise_amplified_results])

    @property
    def noise_amplified_variances(self) -> ndarray:
        return asarray([result.variance for result in self.noise_amplified_results])

    @property
    def shots(self) -> ndarray:
        return asarray([result.shots for result in self.noise_amplified_results])

    @property
    def total_shots(self) -> float:
        return sum(result.shots for result in self.noise_amplified_results)

    @property
    def depths(self) -> ndarray:
        return asarray([result.qc.depth() for result in self.noise_amplified_results])


# Zero-noise extrapolation class:
class ZeroNoiseExtrapolation:

    def __init__(self, qc: QuantumCircuit, exp_val_func: Callable, backend=None, exp_val_filter=None,
                 noise_model: Union[NoiseModel, dict] = None, n_amp_factors: int = 3, shots: int = 8192,
                 pauli_twirl: bool = False, pass_manager: PassManager = None,
                 save_results: bool = False, experiment_name: str = "", option: dict = None,
                 error_controlled_sampling: bool = False, max_shots: int = 2048*8192, error_tol: float = 0.01):
        """
        CONSTRUCTOR

        Parameters
        ----------
        qc : qiskit.QuantumCircuit
            A complete quantum circuit, with measurements, that we want to perform quantum error mitigation on.
            When run on a quantum backend, the circuit should output a set of measurements results from which the
            desired expectation value can be estimated.

        exp_val_func : Callable
            A function that computes the desired expectation value, and its variance, based on the measurement results
            outputted by an execution of a quantum circuit.
            The function should take two arguments: a list of qiskit.result.result.ExperimentResult objects as its first
            argument, and a possible filter as its second.
            The function should return: A numpy.ndarray of expectation values corresponding to each ExperimentResult,
            and a numpy.ndarray of variances in similar fashion.

        backend : A valid qiskit backend, IBMQ device or simulator
            A qiskit backend, either an IBMQ quantum backend or a simulator backend, for circuit executions.
            If none is passed, the qasm_simulator will be used.

        exp_val_filter : any, (optional)
            Optional filter that is passed to the exp_val_func expectation value function.

        noise_model : qiskit.providers.aer.noise.NoiseModel or dict, (optional)
            Custom noise model for circuit executions with the qasm simulator backend.

        n_amp_factors : int, (default set to 3)
            The number of noise amplification factors to be used. For n number of amplification factors, the specific
            noise amplification factors will be [1, 3, 5, ..., 2*n - 1]. Larger amounts of noise amplification factors
            tend to give better results, but slower convergence thus requiring large amounts of shots.
            Higher noise amplification also increases circuit depth, scaling linearly with the amplification factor c_i,
            and at some point the circuit depth and the consecutive decoherence will eliminate any further advantage.

        shots: int, (default set to 8192)
            The number of "shots" of each experiment to be executed, where one experiment is a single execution of a
            quantum circuit. To obtain an error mitigated expectation value, a total of shots*n_amp_factors experiments
            is performed.

        pauli_twirl : bool, (optional)
            Perform Pauli twirling of each noise amplified circuit, True / False.

        pass_manager: qiskit.transpiler.PassManager, (optional)
            Optional custom pass_manager for circuit transpiling. If none is passed, the circuit will be transpiled
            using the qiskit optimization_level=3 preset, which is the heaviest optimization preset.

        save_results: bool, (optional)
            If True, will attempt to read transpiled circuit and experiment results for each noise amplified from disk,
            and if this fails, the transpiled circuit and/or experiment measurement results will be saved to disk.

        experiment_name: string, (optional)
            The experiment name used when reading and writing transpiled circuits and measurement results to disk.
            The experiment name will form the base for the full filename for each written/read file.
            This argument is required if save_result = True.

        option: dict, (optional)
            Options for the writing/reading of transpiled circuits and measurement results.
            option["directory"] gives the directory in which files will be written to/attempted to be read from.
            If no option is passed, the default directory used will be option["directory"] = "results".

        """

        # Set backend for circuit execution. If none is passed, use the qasm_simulator backend.
        if backend is None:
            self.backend = Aer.get_backend("qasm_simulator")
            self.is_simulator = True
        else:
            self.backend = backend
            self.is_simulator = backend.configuration().simulator

        self.exp_val_func = exp_val_func
        self.exp_val_filter = exp_val_filter

        self.noise_model = noise_model

        self.n_amp_factors = n_amp_factors
        self.gamma_coefficients = self.compute_extrapolation_coefficients(n_amp_factors=self.n_amp_factors)
        self.noise_amplification_factors = asarray([(2*i + 1) for i in range(0, self.n_amp_factors)])

        self.pauli_twirl = pauli_twirl

        # Total number of shots
        self.shots = shots

        # variables involved in error controlled sampling:
        self.error_controlled_sampling = error_controlled_sampling
        self.max_shots = max_shots
        self.error_tol = error_tol

        # Variables involved in writing and reading results to/from disk
        self.save_results, self.option = save_results, option
        if self.option is None:
            self.option = {}
        self.experiment_name = ""
        if self.save_results:
            self.set_experiment_name(experiment_name)
            self.create_directory()

        # Initial transpiling of the quantum circuit. If no custom pass manager is passed, the optimization_level=3
        # qiskit preset (the heaviest optimization preset) will be used. If save_results=True, will attempt to read
        # the transpiled circuit from disk.
        circuit_read_from_file = False
        if self.save_results:
            qc_from_file = self.read_from_file(self.experiment_name + ".circuit")
            if not (qc_from_file is None):
                circuit_read_from_file = True
                self.qc = qc_from_file
        if not circuit_read_from_file:
            self.qc = self.transpile_circuit(qc, custom_pass_manager=pass_manager)

        """
        --- Initialization of other variables for later use:
        """

        self.noise_amplified_results = empty((self.n_amp_factors,), dtype=NoiseAmplifiedResult)

        self.result = None

    @staticmethod
    def partition_shots(tot_shots: int) -> (int, int):
        """
        IBMQ devices limits circuit executions to a max of 8192 shots per experiment. To perform more than 8192 shots,
        the experiment has to be partitioned into a set of circuit executions, each with less than 8192 shots.
        Therefore, if shots > 8192, we partition the execution into several repeats of less than 8192 shots each.

        Parameters
        ----------
        tot_shots : int
            The total number of circuit execution shots.

        Returns
        -------
        shots, repeats: (int, int)
            Shots per repeat, number of repeats
        """
        if tot_shots <= 8192:
            return tot_shots, 1
        else:
            if tot_shots % 8192 == 0:
                repeats = (tot_shots // 8192)
            else:
                repeats = (tot_shots // 8192) + 1
            return int(tot_shots / repeats), repeats

    # We use the @property decorator for certain useful data that we might want to retrieve that are stored within the
    # noise amplified results

    @property
    def bare_exp_val(self) -> float:
        return self.noise_amplified_results[0].exp_val

    @property
    def noise_amplified_exp_vals(self) -> ndarray:
        return asarray([result.exp_val for result in self.noise_amplified_results])

    @property
    def noise_amplified_variances(self) -> ndarray:
        return asarray([result.variance for result in self.noise_amplified_results])

    @property
    def depths(self) -> ndarray:
        return asarray([result.qc.depth() for result in self.noise_amplified_results])

    @property
    def mitigated_exp_val(self) -> float:
        if self.result is None:
            return None
        return self.result.exp_val

    # Functions for computing the Richardson extrapolation

    def extrapolate(self, noise_amplified_exp_vals: ndarray):
        """
        The Richardson extrapolation reads

        E^* = \sum_i gamma_i * E(\lambda_i),

        where the gamma_i-coefficients are computed as a function of the amplification factors by solving a system
        of linear equations, this is done in self.compute_extrapolation_coefficients(), and E(\lambda_i) is the
        i-th noise amplified expectation value, corresponding to the amplification factor \lambda_i.

        Ref: https://doi.org/10.1098/rsta.1911.0009

        Parameters
        ----------
        noise_amplified_exp_vals: numpy.ndarray
            The noise amplified expectation value, in order corresponding to amplification factors 1, 3, 5, ... , 2n-1.

        Returns
        -------
        mitigated_exp_val: float
            The mitigated expectation value, which is the exp val extrapolated to the zero-noise case.
        """
        if self.n_amp_factors == 1:
            return noise_amplified_exp_vals[0]
        if not shape(noise_amplified_exp_vals)[0] == shape(self.gamma_coefficients)[0]:
            raise Exception("Shape mismatch between noise_amplified_exp_vals and gamma_coefficients." +
                            " length={:}".format(shape(noise_amplified_exp_vals)[0]) +
                            " does not match length={:}".format(shape(self.gamma_coefficients)[0]))
        return dot(transpose(noise_amplified_exp_vals), self.gamma_coefficients)[0]

    def compute_extrapolation_coefficients(self, n_amp_factors: int = None) -> ndarray:
        """
        Compute the gamma_i-coefficients used in the Richardson extrapolation. We assume the specific noise
        amplification factors to be 1, 3, 5, ..., 2n-1, where n=n_amp_factors is the number of noise amplification
        factors.

        Parameters
        ----------
        n_amp_factors: int
            Number of amplification factors.

        Returns
        ----------
        gamma_coefficients: numpy.ndarray
            The set of coefficients to be used in the Richardson extrapolation.
        """
        if n_amp_factors is None:
            n_amp_factors = self.n_amp_factors
        if n_amp_factors == 1:
            return asarray([1])

        amplification_factors = asarray([2*i + 1 for i in range(n_amp_factors)])

        A, b = zeros((n_amp_factors, n_amp_factors)), zeros((n_amp_factors, 1))

        A[0, :], b[0] = 1, 1

        for k in range(1, n_amp_factors):
            A[k, :] = amplification_factors**k

        gamma_coefficients = solve(A, b)

        return gamma_coefficients

    # Functions involved in saving and loading results from disk

    def set_experiment_name(self, experiment_name):
        """
        Construct the experiment name that will form the base for the filenames that will be read from / written to
        when save_results=True. The full experiment name will contain information about the backend, number of shots,
        and pauli twirling, to ensure that different experiments using different parameters don't read from the same
        data.

        Parameters
        ----------
        experiment_name : str
            The base for the experiment name that will be used for filenames

        """
        if self.save_results and experiment_name == "":
            raise Exception("experiment_name cannot be empty when writing/reading results from disk is activated.")
        self.experiment_name = experiment_name
        self.experiment_name += "__ZNE_CNOT_REP_"
        self.experiment_name += "_backend" + self.backend.name()
        self.experiment_name += "_errorctrld" + str(self.error_controlled_sampling)
        if self.error_controlled_sampling:
            self.experiment_name += "_tol" + str(self.error_tol)
            self.experiment_name += "_maxshots" + str(self.max_shots)
        else:
            self.experiment_name += "_shots" + str(self.shots)
        self.experiment_name += "_paulitwirl" + str(self.pauli_twirl)

    def create_directory(self):
        """
        Attempt to create the directory in which to read from/write to files. The case whereby the directory already
        exists is handled by expection handling.

        The directory is given by self.option["directory"], with the default being "results".

        """
        if not self.save_results:
            return
        directory = self.option.get("directory", "results")
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise e

    def read_from_file(self, filename: str):
        """
        Attempts to read data for a given filename, looking in the directory given by self.option["directory"].

        Parameters
        ----------
        filename : str
            The full filename for the file in question

        Returns
        -------
        data : any
            The data read from said file. None if the file wasn't found
        """
        directory = self.option.get("directory", "results")
        if os.path.isfile(directory + "/" + filename):
            file = open(directory + "/" + filename, "rb")
            data = pickle.load(file)
            file.close()
            return data
        else:
            return None

    def write_to_file(self, filename: str, data):
        """
        Writes data to file with given filename, located in the directory given by self.option["directory"].

        Parameters
        ----------
        filename : str
            The full filename of the file to be written to.
        data : any
            The data to be stored.

        """
        directory = self.option.get("directory", "results")
        file = open(directory + "/" + filename, "wb")
        pickle.dump(data, file)
        file.close()

    # Functions for processing and executing the quantum circuits

    def noise_amplify_and_pauli_twirl_cnots(self, qc: QuantumCircuit, amp_factor: int,
                                            pauli_twirl: bool = False) -> QuantumCircuit:
        """
        Amplify CNOT-noise by extending each CNOT-gate as CNOT^amp_factor and possibly Pauli-twirl all CNOT-gates

        Using CNOT*CNOT = I, the identity, and an amp_factor = (2*n + 1) for an integer n, then the
        extended CNOT will have the same action as a single CNOT, but with the noise amplified by
        a factor amp_factor. This thus method allows for circuit-level noise amplification.

        For efficiency, this function does both noise amplification and pauli twirling (optionally) at the same time.
        Separate functions for noise amplification and for pauli twirling are included at the end of this file, for
        completeness.

        :param qc: Quantum circuit for which to Pauli twirl all CNOT gates and amplify CNOT-noise
        :param amp_factor: The noise amplification factor, must be (2n + 1) for n = 0,1,2,3,...
        :param pauli_twirl: Add pauli twirling True / False
        :return: Noise-amplified and possibly Pauli-twirled Quantum Circuit
        """

        if (amp_factor - 1) % 2 != 0:
            raise Exception("Invalid amplification factors", amp_factor)

        # The circuit may be expressed in terms of various types of gates.
        # The 'Unroller' transpiler pass 'unrolls' (decomposes) the circuit gates to be expressed in terms of the
        # physical gate set [u1,u2,u3,cx]

        # The cz, cy (controlled-Z and -Y) gates can be constructed from a single cx-gate and single-qubit gates.
        # For backends with native gate sets consisting of some set of single-qubit gates and either the cx, cz or cy,
        # unrolling the circuit to the ["u3", "cx"] basis, amplifying the cx-gates, then unrolling back to the native
        # gate set and doing a single-qubit optimization transpiler pass, is thus still general.

        unroller_ugatesandcx = Unroller(["u1", "u2", "u3", "cx"])
        pm = PassManager(unroller_ugatesandcx)

        unrolled_qc = pm.run(qc)

        circuit_qasm = unrolled_qc.qasm()
        new_circuit_qasm_str = ""

        qreg_name = find_qreg_name(circuit_qasm)

        for i, line in enumerate(circuit_qasm.splitlines()):
            if line[0:2] == "cx":
                for j in range(amp_factor):
                    if pauli_twirl:
                        new_circuit_qasm_str += pauli_twirl_cnot_gate(qreg_name, line)
                    else:
                        new_circuit_qasm_str += (line + "\n")
            else:
                new_circuit_qasm_str += line + "\n"

        new_qc = QuantumCircuit.from_qasm_str(new_circuit_qasm_str)

        # The "Optimize1qGates" transpiler pass optimizes adjacent single-qubit gates, for a native gate set with the
        # u3 gates it collapses any chain of adjacent single-qubit gates into a single, equivalent u3-gate.
        # We want to collapse unnecessary single-qubit gates to minimize circuit depth, but not CNOT-gates
        # as these give us the noise amplification.
        unroller_backendspecific = Unroller(self.backend.configuration().basis_gates)
        optimize1qates = Optimize1qGates()

        pm = PassManager([unroller_backendspecific, optimize1qates])

        return pm.run(new_qc)

    def transpile_circuit(self, qc: QuantumCircuit, custom_pass_manager: PassManager = None) -> QuantumCircuit:
        """
        Transpile and optimize the input circuit, optionally by  using a custom pass manager.
        If no custom pass manager is given, the optimization_level = 3 preset for the qiskit transpiler,
        the heaviest optimization preset, will be used.

        As we want to add additional CNOTs for noise amplification and possibly additional single qubit gates
        for Pauli twirling, we need to transpile the circuit before both the noise amplification is applied and
        before circuit execution. This is to avoid the additional CNOT-gates beinng removed by the transpiler.

        The Optimize1qGates transpiler pass will be used later to optimize single qubit gates added during
        the Pauli-twirling, as well as the Unroller pass which merely decomposes the given circuit gates into
        the given set of basis gates.

        Parameters
        ----------
        qc : qiskit.QuantumCircuit
            The original bare quantum circuit.
        custom_pass_manager : qiskit.transpiler.PassManager, (optional)
            A custom pass manager to be used in transpiling.

        Returns
        -------
        transpiled_circuit : qiskit.QuantumCircuit
            The transpiled quantum circuit.
        """

        if custom_pass_manager is None:
            pass_manager_config = PassManagerConfig(basis_gates=["id", "u1", "u2", "u3", "cx"],
                                                    backend_properties=self.backend.properties())
            if not self.is_simulator:
                pass_manager_config.coupling_map = CouplingMap(self.backend.configuration().coupling_map)

            pass_manager = level_3_pass_manager(pass_manager_config)
        else:
            pass_manager = custom_pass_manager

        self.passes = pass_manager.passes()  # Saves the list of passes used for transpiling

        transpiled_circuit = pass_manager.run(qc)

        # As there is some randomness involved in the qiskit transpiling we might want to save
        # the specific transpiled circuit that is used in order to access it later.
        if self.save_results:
            filename = self.experiment_name + ".circuit"
            self.write_to_file(filename, transpiled_circuit)

        return transpiled_circuit

    def execute_circuit(self, qc: QuantumCircuit, shots=None) -> Result:
        """
        Execute a single experiment consisting of the execution of a quantum circuit over a specified number of shots.
        One experiment may need to be partitioned into a set of several identical circuit executions. This is due to the
        IBMQ quantum devices limiting circuit executions to a maximum of 8192 shots per.

        Parameters
        ----------
        qc : qiskit.QuantumCircuit
            The specific quantum circuit to be executed.
        shots : int, (optional)
            The number of shots of the circuit execution. If none is passed, self.shots is used.
        Returns
        -------
        circuit_measurement_results : qiskit.result.result.Result
            A Result object containing the data and measurement results for the circuit executions.
        """

        if shots is None:
            shots, repeats = self.partition_shots(self.shots)
        else:
            shots, repeats = self.partition_shots(shots)

        # The max number of shots on a single execution on the IBMQ devices is 8192.
        # If shots > 8192, we have to partition the execution into several sub-executions.
        # Note that several circuits can be entered into the IBMQ queue at once by passing them in a list.
        execution_circuits = [qc.copy() for i in range(repeats)]

        # non-simulator backends throws "unexpected argument" exception when passing noise_model argument to them
        if self.is_simulator:
            job = execute(execution_circuits, backend=self.backend, noise_model=self.noise_model,
                          pass_manager=PassManager(), shots=shots)
        else:
            job = execute(execution_circuits, backend=self.backend,
                          pass_manager=PassManager(), shots=shots)

        circuit_measurement_results = job.result()

        return circuit_measurement_results

    # Functions involved in computing the noise amplified and mitigated expectation values and related measures.

    def compute_exp_val(self, result: Result) -> (float, float, ndarray):
        """
        Compute the expectation value and variance for a set of circuit executions. We assume that all separate circuit
        execution was run with the same number of shots.

        Parameters
        ----------
        result : qiskit.result.result.Result
            A qiskit Result object containing all measurement results from a set of quantum circuit executions.

        Returns
        -------
        averaged_experiment_exp_vals, averaged_experiment_variances experiment_exp_vals : Tuple[float, numpy.ndarray]
            The final estimated experiment expectation value and variance, averaged over all circuit sub-executions,
            and a numpy array containing the expectation values for each circuit sub-execution.
        """

        experiment_results = result.results

        experiment_exp_vals, experiment_variances = self.exp_val_func(experiment_results, self.exp_val_filter)

        return average(experiment_exp_vals), average(experiment_variances), asarray(experiment_exp_vals)

    def compute_error_controlled_exp_val(self, noise_amplified_qc: QuantumCircuit, gamma_coeff: float, shots: int = None,
                                         conf_index: int = 2, verbose: bool = False) -> (float, float, float):
        """
        Handles optional error controlled sampling of the quantum circuit. Returns the resulting estimated expectation
        value and variance, and the total number of shots used.

        Parameters
        ----------
        noise_amplified_qc: qiskit.QuantumCircuit
        gamma_coeff: float
        shots: int
        conf_index: int
        verbose: bool

        Returns
        -------
        exp_val, variance, shots: float, float, int

        """

        if shots is None:
            shots = self.shots

        result = self.execute_circuit(qc=noise_amplified_qc, shots=shots)

        exp_val, variance, _ = self.compute_exp_val(result)

        if not self.error_controlled_sampling:
            return exp_val, variance, shots

        if verbose:
            print("Error controlled sampling, " +
                  "min_shots={:}, max_shots={:}, error_tol={:}, conf_index={:}".format(shots, self.max_shots,
                                                                                       self.error_tol, conf_index))

        total_shots = int(self.n_amp_factors * (gamma_coeff**2) * (conf_index**2) * (variance / self.error_tol**2))

        if verbose:
            print("Variance={:.4f}, gamma_coeff={:} need a total of {:} shots for convergence.".format(variance,
                                                                                                       gamma_coeff,
                                                                                                       total_shots))

        if total_shots <= shots:
            return exp_val, variance, shots
        elif total_shots > self.max_shots:
            total_shots = self.max_shots

        new_shots = int(total_shots - shots)

        if verbose:
            print("Executing {:} additional shots.".format(new_shots))

        result = self.execute_circuit(qc=noise_amplified_qc, shots=new_shots)

        new_exp_val, new_variance, _ = self.compute_exp_val(result)

        error_controlled_exp_val = (shots/total_shots) * exp_val + (new_shots/total_shots) * new_exp_val
        error_controlled_variance = (shots/total_shots) * variance + (new_shots/total_shots) * new_variance

        return error_controlled_exp_val, error_controlled_variance, total_shots

    def compute_noise_amplified_exp_val(self, amp_factor, gamma_coeff, verbose: bool = False):
        """
        Creates the noise amplified circuit for a given amplification factors and computes the corresponding estimated
        expectation value and variance.

        Parameters
        ----------
        amp_factor: int
        gamma_coeff: float
        verbose: bool

        Returns
        -------
        noise_amplified_result: NoiseAmplifiedResult

        """
        noise_amplified_qc = self.noise_amplify_and_pauli_twirl_cnots(qc=self.qc, amp_factor=amp_factor,
                                                                      pauli_twirl=self.pauli_twirl)

        if verbose:
            print("Circuit created. Depth = {:}. Executing.".format(noise_amplified_qc.depth()))

        exp_val, variance, total_shots = self.compute_error_controlled_exp_val(noise_amplified_qc,
                                                                               gamma_coeff=gamma_coeff,
                                                                               shots=self.shots, verbose=verbose)

        noise_amplified_result = NoiseAmplifiedResult(amp_factor=amp_factor, shots=total_shots,
                                                      qc=noise_amplified_qc, depth=noise_amplified_qc.depth(),
                                                      exp_val=exp_val, variance=variance)
        return noise_amplified_result

    def estimate_error(self) -> float:
        """
        Estimate the error in the mitigated expectation value based on the variance found in the noise amplified circuit
        executions.

        Returns
        -------
        error: float
            Estimated error in the mitigated expectation value.
        """
        error = 0
        for i, amp_res in enumerate(self.noise_amplified_results):
            variance, shots = amp_res.variance, amp_res.shots
            gamma_coeff = self.gamma_coefficients[i]

            error += gamma_coeff**2 * (variance / shots)

        error = sqrt(error)

        return error

    def mitigate(self, verbose: bool = False) -> float:
        """
        Perform the full quantum error mitigation procedure.

        Parameters
        ----------
        verbose : bool
            Do prints throughout the computation.

        Returns
        -------
        result : float
            The mitigated expectation value.
        """

        for i, amp_factor in enumerate(self.noise_amplification_factors):

            if verbose:
                print("Amplification factor = {:}.".format(amp_factor))

            noise_amplified_result, result_read_from_file = None, False

            # If self.save_results=True, attempt to read noise amplified result from disk
            if self.save_results:
                temp = self.read_from_file(filename=self.experiment_name + "_r{:}.result".format(amp_factor))
                if (temp is not None) and (type(temp) is NoiseAmplifiedResult):
                    noise_amplified_result = temp
                    result_read_from_file = True
                    if verbose:
                        print("Noise amplified result successfully read from disk.")
                else:
                    if verbose:
                        print("Tried to read results from disk, but results were not found.")

            gamma_coeff = self.gamma_coefficients[i]

            if not result_read_from_file:
                noise_amplified_result = self.compute_noise_amplified_exp_val(amp_factor=amp_factor,
                                                                              gamma_coeff=gamma_coeff,
                                                                              verbose=verbose)

                if self.save_results:
                    self.write_to_file(filename=self.experiment_name + "_r{:}.result".format(amp_factor),
                                       data=noise_amplified_result)
                    if verbose:
                        "Noise amplified result successfully written to disk."

            self.noise_amplified_results[i] = noise_amplified_result

            if verbose:
                print("Noise amplified exp val = {:.8f}, ".format(noise_amplified_result.exp_val) +
                      "variance = {:.8f}, ".format(noise_amplified_result.variance) +
                      "total shots executed = {:}.".format(noise_amplified_result.shots))

        mitigated_exp_val = self.extrapolate(self.noise_amplified_exp_vals)

        self.result = ZeroNoiseExtrapolationResult(qc=self.qc,
                                                   noise_amplified_results=self.noise_amplified_results,
                                                   noise_amplification_factors=self.noise_amplification_factors,
                                                   gamma_coefficients=self.gamma_coefficients,
                                                   exp_val=mitigated_exp_val
                                                   )

        if verbose:
            print("-----\nERROR MITIGATION DONE\n" +
                  "Bare circuit expectation value: {:.8f}\n".format(self.result.bare_exp_val) +
                  "Noise amplified expectation values: {:}\n".format(self.result.noise_amplified_exp_vals) +
                  "Circuit depths: {:}\n".format(self.result.depths) +
                  "-----\n" +
                  "Mitigated expectation value: {:.8f}\n".format(self.result.exp_val))

        return mitigated_exp_val

"""
--- PAULI TWIRLING AND NOISE AMPLIFICATION HELP FUNCTIONS
"""

# Conversion from pauli x/y/z-gates to physical u1/u3-gates in correct OpenQASM-format
PHYSICAL_GATE_CONVERSION = {"X": "u3(pi,0,pi)", "Z": "u1(pi)", "Y": "u3(pi,pi/2,pi/2)"}


def find_qreg_name(circuit_qasm: str) -> str:
    """
    Finds the name of the quantum register in the circuit. Assumes a single quantum register.

    Parameters
    ----------
    circuit_qasm : str
        The OpenQASM-string for the circuit

    Returns
    -------
    qreg_name :str
        The name of the quantum register
    """
    for line in circuit_qasm.splitlines():
        if line[0:5] == "qreg ":
            qreg_name = ""
            for i in range(5, len(line)):
                if line[i] == "[" or line[i] == ";":
                    break
                elif line[i] != " ":
                    qreg_name += line[i]
            return qreg_name


def find_cnot_control_and_target(qasm_line: str) -> (int, int):
    """
    Find the indices of the control and target qubits for a specific CNOT-gate.

    Parameters
    ----------
    qasm_line : str
        The line containing the CNOT-gate in question taken from the OpenQASM-format string of the quantum circuit.

    Returns
    -------
    control, target : Tuple[int, int]
        qubit indices for control and target qubits
    """
    qubits = []
    for i, c in enumerate(qasm_line):
        if c == "[":
            qubit_nr = ""
            for j in range(i + 1, len(qasm_line)):
                if qasm_line[j] == "]":
                    break
                qubit_nr += qasm_line[j]
            qubits.append(int(qubit_nr))
    return qubits[0], qubits[1]


def propagate(control_in: str, target_in: str):
    """
    Finds the c,d gates such that (a x b) CNOT (c x d) = CNOT for an ideal CNOT-gate, based on the a (control_in)
    and b (target_in) pauli gates by "propagating" the a,b gates over a CNOT-gate by the following identities:

    (X x I) CNOT = CNOT (X x X)
    (I x X) CNOT = CNOT (I x X)
    (Z x I) CNOT = CNOT (I x Z)
    (I x Z) CNOT = XNOT (Z x Z)

    Note that instead of Pauli-twirling with [X,Z,Y] we use [X,Z,XZ] where XZ = -i*Y.
    The inverse of XZ is ZX = -XZ = i*Y. The factors of plus minus i are global phase factors which can be ignored.

    Parameters
    ----------
    control_in : str
        The Pauli operator on control qubit before the CNOT, i.e., a
    target_in : str
        The Pauli operator on target qubit before the CNOT, i.e., b
    Returns
    -------
    control_out, target_out : Tuple[str, str]
        The operators c and d such that (a x b) CNOT (c x d) = CNOT
    """

    control_out, target_out = '', ''
    if 'X' in control_in:
        control_out += 'X'
        target_out += 'X'
    if 'X' in target_in:
        target_out += 'X'
    if 'Z' in control_in:
        control_out += 'Z'
    if 'Z' in target_in:
        control_out += 'Z'
        target_out += 'Z'

    # Pauli gates square to the identity, i.e. XX = I, ZZ = I
    # Remove all such occurences from the control & target out Pauli gate strings
    if 'ZZ' in control_out:
        control_out = control_out[:-2]
    if 'ZZ' in target_out:
        target_out = target_out[:-2]
    if 'XX' in control_out:
        control_out = control_out[2:]
    if 'XX' in target_out:
        target_out = target_out[2:]

    # If no Pauli gates remain then we have the identity gate I
    if control_out == '':
        control_out = 'I'
    if target_out == '':
        target_out = 'I'

    # The inverse of XZ is ZX, therefore we reverse the gate order to obtain the correct pauli gates c,d
    # such that (a x b) CNOT (c x d) = CNOT (c^-1 x d^-1) (c x d) = CNOT
    return control_out[::-1], target_out[::-1]


def apply_qasm_pauli_gate(qreg_name: str, qubit: int, pauli_gates: str):
    """
    Construct an OpenQASM-string line with the given Pauli-gates applid to the given qubit.

    Parameters
    ----------
    qreg_name :str
        The name of the qiskit.QuantumRegister containing the qubit.
    qubit : int
        The index of the qubit.
    pauli_gates : str
        A string determining the Pauli-gates to be applied. Must be a sequence the characters I, X, Y and/or Z.
    Returns
    -------
    new_qasm_line : str
        An OpenQASM string with the Pauli-gates applied to the qubit.
    """
    new_qasm_line = ''
    for gate in pauli_gates:
        if gate != 'I':
            if gate not in PHYSICAL_GATE_CONVERSION.keys():
                raise Exception("Invalid Pauli-gate used in Pauli-twirl: {:}".format(gate))
            u_gate = PHYSICAL_GATE_CONVERSION[gate]
            new_qasm_line += u_gate + ' ' + qreg_name + '[' + str(qubit) + '];' + '\n'
    return new_qasm_line


def pauli_twirl_cnot_gate(qreg_name: str, qasm_line_cnot: str) -> str:
    """
    Pauli-twirl a specific CNOT-gate. This involves drawing two random Pauli-gates a and b, picked from the single-qubit
    Pauli set {Id, X, Y, Z}, then determining the corresponding two Pauli-gates c and d such that
    (a x b) * CNOT * (c x d) = CNOT, for an ideal CNOT.

    The original CNOT gates is then replaced by the gate ((a x b) * CNOT * (c x d)). This transforms the noise in the
    CNOT-gate into stochastic Pauli-type noise. An underlying assumption is that the noise in the single-qubit Pauli
    gates is negligible to the noise in the CNOT-gates.

    Parameters
    ----------
    qreg_name : str
        The name of the qiskit.QuantumRegister for the qubits in question.
    qasm_line_cnot : str
        The OpenQASM-string line containing the CNOT-gate.

    Returns
    -------
    new_qasm_line : str
        A new OpenQASM-string section to replace the aforementioned OpenQASM line containing the CNOT-gate, where not
        the CNOT-gate has been Pauli-twirled.
    """

    control, target = find_cnot_control_and_target(qasm_line_cnot)

    # Note: XZ = -i*Y, with inverse (XZ)^-1 = ZX = i*Y. This simplifies the propagation of gates a,b over the CNOT
    pauli_gates = ["I", "X", "Z", "XZ"]

    a = random.choice(pauli_gates)
    b = random.choice(pauli_gates)

    # Find gates such that:
    # (a x b) CNOT (c x d) = CNOT for an ideal CNOT-gate,
    # by propagating the Pauli gates through the CNOT

    c, d = propagate(a, b)

    new_qasm_line = apply_qasm_pauli_gate(qreg_name, control, a)
    new_qasm_line += apply_qasm_pauli_gate(qreg_name, target, b)
    new_qasm_line += qasm_line_cnot + '\n'
    new_qasm_line += apply_qasm_pauli_gate(qreg_name, target, d)
    new_qasm_line += apply_qasm_pauli_gate(qreg_name, control, c)

    return new_qasm_line


def pauli_twirl_cnots(qc: QuantumCircuit) -> QuantumCircuit:
    """
    Pauli-twirl all CNOT-gates in a general quantum circuit. This function is included here for completeness.

    Parameters
    ----------
    qc : qiskit.QuantumCircuit
        The original quantum circuit.

    Returns
    -------
    pauli_twirled_qc : qiskit.QuantumCircuit
        The quantum circuit where all CNOT-gates have been Pauli-twirled.
    """

    # The circuit may be expressed in terms of various types of gates.
    # The 'Unroller' transpiler pass 'unrolls' the circuit to be expressed in terms of the
    # physical gate set [u1,u2,u3,cx]
    unroller = Unroller(["u1", "u2", "u3", "cx"])
    pm = PassManager(unroller)

    unrolled_qc = pm.run(qc)

    circuit_qasm = unrolled_qc.qasm()
    new_circuit_qasm_str = ""

    qreg_name = find_qreg_name(circuit_qasm)

    for i, line in enumerate(circuit_qasm.splitlines()):
        if line[0:2] == "cx":
            new_circuit_qasm_str += pauli_twirl_cnot_gate(qreg_name, line)
        else:
            new_circuit_qasm_str += line + "\n"

    new_qc = QuantumCircuit.from_qasm_str(new_circuit_qasm_str)

    # The "Optimize1qGates" transpiler pass optimizes chains of single-qubit gates by collapsing them into
    # a single, equivalent u3-gate

    # We want to avoid that the transpiler optimizes CNOT-gates, as the ancillary CNOT-gates must be kept
    # to keep the noise amplification

    optimize1qates = Optimize1qGates()
    pm = PassManager(optimize1qates)

    return pm.run(new_qc)

def noise_amplify_cnots(qc: QuantumCircuit, amp_factor: int):
    """
    Noise amplify all CNOT-gates in the given QuantumCircuit by expanding each CNOT-gate as CNOT -> CNOT^a, where a is
    the noise amplification factor, a = 2*n - 1. Included here for completeness.

    Parameters
    ----------
    qc: qiskit.QuantumCircuit
        Quantum circuit to be noise amplified
    amp_factor:
        The noise amplification factor. Must be odd
    Returns
    -------
    noise_amplified_qc: qiskit.QuantumCircuit
        The noise amplified circuit
    """

    if (amp_factor - 1) % 2 != 0:
        raise Exception("Invalid amplification factors", amp_factor)

    # The circuit may be expressed in terms of various types of gates.
    # The 'Unroller' transpiler pass 'unrolls' (decomposes) the circuit gates to be expressed in terms of the
    # physical gate set [u1,u2,u3,cx]

    # The cz, cy (controlled-Z and -Y) gates can be constructed from a single cx-gate and single-qubit gates.
    # For backends with native gate sets consisting of some set of single-qubit gates and either the cx, cz or cy,
    # unrolling the circuit to the ["u3", "cx"] basis, amplifying the cx-gates, then unrolling back to the native
    # gate set and doing a single-qubit optimization transpiler pass, is thus still general.

    unroller_ugatesandcx = Unroller(["u", "cx"])
    pm = PassManager(unroller_ugatesandcx)

    unrolled_qc = pm.run(qc)

    circuit_qasm = unrolled_qc.qasm()
    new_circuit_qasm_str = ""

    # qreg_name = find_qreg_name(circuit_qasm)

    for i, line in enumerate(circuit_qasm.splitlines()):
        if line[0:2] == "cx":
            for j in range(amp_factor):
                new_circuit_qasm_str += line + "\n"
        else:
            new_circuit_qasm_str += line + "\n"

    new_qc = QuantumCircuit.from_qasm_str(new_circuit_qasm_str)

    return new_qc

