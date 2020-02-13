import re #because we need regular expressions
import numpy as np #because we need the random number generator
from qiskit import *
from qiskit.tools.monitor import job_monitor

import sys
sys.path.append('../')

from qiskit_utilities.utilities import *

import numpy as np

Pauligateset=['id','x','y','z']

def equal(A, B, tolerance=1E-6, allowPhaseCheck=True):
    """
    Returns True if A and B are equal within tolerance tol, element-wise
    :param A: nxm matrix
    :param B: nxm matrix
    :param tolerance: Tolerance
    :param allowPhaseCheck: True if A and B should be considered equal if they are equal up to a complex phase factor
    :raises ValueError: if A, B not correct shape
    :return: True or False
    """
    dim0 = A.shape[0]
    dim1 = A.shape[1]
    if len(A.shape) > 2 or len(B.shape) > 2:
        raise ValueError("Both arguments must be n x m matrices.")
    if dim0 != B.shape[0] or dim1 != B.shape[1]:
        raise ValueError("Arguments must be of equal dimension.")

    checkPhase = False

    for ind in range(dim0):
        if checkPhase:
            break
        for j in range(dim1):
            num = np.absolute(A[ind, j] - B[ind, j])
            if num > tolerance:
                if allowPhaseCheck:
                    checkPhase = True  # There might be a (global) phase difference between the matrices
                else:
                    return False
                break

    if checkPhase:# Handle the case A = e^(i * theta) * B
        first = True
        eiPhase = 1

        for ind in range(dim0):
            for j in range(dim1):
                A_ij = A[ind, j]
                B_ij = B[ind, j]

                if np.absolute(A_ij) < tolerance:
                    A_ij = 0
                if np.absolute(B_ij) < tolerance:
                    B_ij = 0

                if A_ij == 0 and B_ij == 0:
                    continue
                elif (A_ij == 0 and B_ij != 0) or (A_ij != 0 and B_ij == 0):
                    # If one is zero, multiplying by phase factor does not change anything.
                    # Therefore, the other must be zero for the matrices to be equal up to a phase factor.
                    return False

                rel = A_ij / B_ij
                if np.absolute(np.absolute(rel) - 1) > tolerance:  # a/b must be 1.000... * e^(i * theta)
                    return False
                if first:
                    first = False
                    eiPhase = rel  # e ^(i * theta)
                if np.absolute(eiPhase - rel) > tolerance:
                    return False
    return True

def getPaulitwirlingPairsCX(printpairs=False):
    ID1q = np.array([[1, 0], [0, 1]], dtype=np.complex_)
    sigmaX = np.array([[0, 1], [1, 0]], dtype=np.complex_)
    sigmaY = np.array([[0, 0 - 1.0j], [0 + 1.0j, 0]], dtype=np.complex_)
    sigmaZ = np.array([[1, 0], [0, -1]], dtype=np.complex_)

    ID2q = np.kron(ID1q, ID1q)

    tol = 1E-8

    paulis = np.array([ID1q, sigmaX, sigmaY, sigmaZ], dtype=np.complex_)

    cX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex_)

    pairs = np.zeros((4, 4, 2), dtype=int)

    for a in range(4):  # sigma_control ^a
        pauli_ca = paulis[a]
        for b in range(4):  # sigma_target ^b
            pauli_tb = paulis[b]
            for c in range(4):  # sigma_control ^c
                pauli_cc = paulis[c]
                for d in range(4):  # sigma_target ^d
                    pauli_td = paulis[d]
                    LHS = np.kron(pauli_cc, pauli_td)
                    RHS = cX @ np.kron(pauli_ca, pauli_tb) @ cX.transpose().conjugate()
                    if equal(LHS, RHS, tol):
                        if printpairs:
                            print(Pauligateset[a],Pauligateset[b],Pauligateset[c],Pauligateset[d])
                        pairs[a][b][0] = c
                        pairs[a][b][1] = d
    return pairs

twirlingPairs = getPaulitwirlingPairsCX()

def create_Paulitwirled_and_noiseamplified_circuit(circuit,r,two_error_map,paulitwirling=True,controlledgatename='cx'):
    '''Pauli-twirl and amplify noise of controlled gates in a circuit

    Args:
        circuit: the original circuit
        r: noise amplification factor
        two_error_map: map of error rates of controlled gate between two qubits
        paulitwirling: turn Pauli-twirling on or off
        controlledgatename: name of the controlled gate to apply Pauli-twirling and error amplification to

    Returns:
        new circuit that is Pauli-twirled and errors are amplified by a factor for r
    '''
    newqasm_str=""
    qs=circuit.qasm()
    qregname=circuit.qregs[0].name
    for line in iter(qs.splitlines()):
        if line.startswith(controlledgatename):
            ## Find the number of the control and the target qubit
            search_results = re.finditer(r'\[.*?\]', line)
            count=0
            for item in search_results:
                if count==0:
                    control_ind=int(item.group(0).lstrip('[').rstrip(']'))
                else:
                    target_ind=int(item.group(0).lstrip('[').rstrip(']'))
                count+=1
            ## Apply Pauli-twirling
            if paulitwirling:
                indices_ab = np.random.randint(0, 4, 2)
                indices_cd = twirlingPairs[indices_ab[0]][indices_ab[1]]

                if indices_ab[0]>0:
                    newqasm_str+=Pauligateset[indices_ab[0]]+" "+qregname+"["+str(control_ind)+"];\n"
                if indices_ab[1]>0:
                    newqasm_str+=Pauligateset[indices_ab[1]]+" "+qregname+"["+str(target_ind)+"];\n"
                    
                newqasm_str+=line+"\n"
                
                if indices_cd[0]>0:
                    newqasm_str+=Pauligateset[indices_cd[0]]+" "+qregname+"["+str(control_ind)+"];\n"
                if indices_cd[1]>0:
                    newqasm_str+=Pauligateset[indices_cd[1]]+" "+qregname+"["+str(target_ind)+"];\n"
            else:
                newqasm_str+=line+"\n"

            ## increase the error rate of a cx gate
            if np.random.uniform(0, 1) < (r - 1) * two_error_map[control_ind][target_ind]:
                ### we need to avoid that the indentity is drawn for the control and the target qubit at the same time
                ### there are 4x4 combinations
                # I I    X I    Y I    Z I
                # I X    X X    Y X    Z X
                # I Y    X Y    Y Y    Z Y
                # I Z    X Z    Y Z    Z Z
                # with global indexation for random numbers
                # 0      4      8      12
                # 1      5      9      13
                # 2      6      10     14
                # 3      7      11     15
                # and local indexation
                # 0,0    1,0    2,0    3,0
                # 0,1    1,1    2,1    3,1
                # 0,2    1,2    2,2    3,2
                # 0,3    1,3    2,3    3,3
                # so we make sure to never draw 0 = (0,0) = I I
                ind_ef = np.random.randint(1, 16, 1)
                ind_e=int(int(ind_ef[0])/4)
                ind_f=ind_ef[0]%4
                if ind_e>0:
                    newqasm_str+=Pauligateset[ind_e]+" "+qregname+"["+str(control_ind)+"];\n"
                if ind_f>0:
                    newqasm_str+=Pauligateset[ind_f]+" "+qregname+"["+str(target_ind)+"];\n"

        else:
            newqasm_str+=line+"\n"
    circ=QuantumCircuit().from_qasm_str(newqasm_str)
    return circ

def Richardson_extrapolate(E, c):
    n=E.shape[0]
    if c.shape[0] != n:
        raise ValueError('E and c must have the same dimension.')
    if n<=1:
        raise ValueError('the dimension of E and c must be larger than 1.')
    A = np.zeros((n,n))
    b = np.zeros((n,1))
    #must sum to 1
    A[0,:] = 1
    b[0] = 1
    for k in range(1,n):
        A[k,:] = c**k
    x=np.linalg.solve(A,b)
    return np.dot(np.transpose(E),x)

def mitigate(circuit, amplification_factors,\
             expectationvalue_fun,\
             execution_backend, \
             experimentname, cx_error_map,\
             num_shots, num_experiments,\
             target_backend=None, noise_model=None, basis_gates=None,\
             paulitwirling=True, verbose=True):
    """
    it is of utter most importance, that the circit is executable on the backend that it is to be executed/targeted for
    target_backend: is used if execution_backend is a simulator
    noisemodel: is used if execution_backend is a simulator
    this function is implemented with convenience in mind, the classical part can be trivially made more memory efficient
    """
    optimization_level=1

    n_qubits = execution_backend.configuration().n_qubits
    is_simulator = execution_backend.configuration().simulator

    max_depth_dict={}
    mean_depth_dict={}
    max_depth_transpiled_dict={}
    mean_depth_transpiled_dict={}
    jobs_dict={}
    E_dict={}
    E_av_dict={}
    result_dict={}

    ### sanity checks
    if len(amplification_factors)<2:
        raise ValueError("specify at least 2 amplification factors, e.g., (1,2) ")
    if is_simulator:
        if target_backend == None:
            raise ValueError("you need to specify a taget backend")
        if noise_model == None:
            raise ValueError("you need to specify a noise model")
        if basis_gates == None:
            raise ValueError("you need to specify basis gates")
    else:
        execution_backend = target_backend

    if verbose:
        print("Sanity checks passed")

    if is_simulator:
        # in the case of a simulator,
        # we do not need to split the runs,
        # because max_experiments is not limited
        experimentname+="_backend"+execution_backend.name()
        experimentname+="_noisemodel"+target_backend.name()
        experimentname+="_shots"+str(num_shots)
        experimentname+="_experiments"+str(num_experiments)
        experimentname+="_paulitwirling"+str(paulitwirling)
        for r in amplification_factors:
            name=experimentname+"_r"+str(r)
            result_dict[name] = read_results(name)
            if verbose:
                if result_dict[name] == None:
                    print("Could not read result for job '",name, "' from disk")
                else:
                    print("Result for job '",name, "' successfully read from disk")

            ### read circuit depth statistics from file
            if not result_dict[name] == None:
                with open('results/'+name+'.mean_circuit_depth','r') as f:
                    mean_depth_dict[name]=float(f.read())
                with open('results/'+name+'.max_circuit_depth','r') as f:
                    max_depth_dict[name]=float(f.read())
                with open('results/'+name+'.mean_transpiled_circuit_depth','r') as f:
                    mean_depth_transpiled_dict[name]=float(f.read())
                with open('results/'+name+'.max_transpiled_circuit_depth','r') as f:
                    max_depth_transpiled_dict[name]=float(f.read())

        for r in amplification_factors:
            name=experimentname+"_r"+str(r)
            if not result_dict[name] == None:
                continue
            mean_depth=0
            max_depth=0
            mean_depth_transpiled=0
            max_depth_transpiled=0
            circuits_r=[]
            for p in range(1,num_experiments+1):
                if verbose and p%25==0:
                    print("Creating circuits for '",name, "'", p, "/",num_experiments, end='\r')
                circ_tmp = create_Paulitwirled_and_noiseamplified_circuit(\
                                    circuit, r, cx_error_map, paulitwirling)
                depth = circ_tmp.depth()
                mean_depth += depth
                max_depth = max(max_depth,depth)
                # now we can transpile to combine single qubit gates, etc.
                circ_tmp_transpiled=transpile(circ_tmp,\
                                              backend=target_backend,\
                                              optimization_level=optimization_level)
                circuits_r.append(circ_tmp_transpiled)
                depth=circ_tmp_transpiled.depth()
                mean_depth_transpiled += depth
                max_depth_transpiled = max(max_depth_transpiled,depth)
            if verbose:
                print("Creating circuits for '",name, "'", num_experiments, "/",num_experiments)
            max_depth_dict[name]=max_depth
            mean_depth_dict[name]=mean_depth/num_experiments
            max_depth_transpiled_dict[name]=max_depth_transpiled
            mean_depth_transpiled_dict[name]=mean_depth_transpiled/num_experiments
            if verbose:
                print("Starting job for '",name, "'")
            jobs_dict[name] = execute(circuits_r,\
                            execution_backend,\
                            noise_model=noise_model,\
                            basis_gates=basis_gates,\
                            shots=num_shots)

        for r in amplification_factors:
            name=experimentname+"_r"+str(r)
            if not result_dict[name] == None:
                continue
            job_monitor(jobs_dict[name])
            success = write_results(name,jobs_dict[name])
            if verbose:
                if success:
                    print("Result for job '",name, "' successfully written to disk")
                else:
                    print("Could not write result for job '",name, "' from disk")

            ### write circuit depth statistics to file
            with open('results/'+name+'.mean_circuit_depth','w') as f:
                f.write(str(mean_depth_dict[name]))
            with open('results/'+name+'.max_circuit_depth','w') as f:
                f.write(str(max_depth_dict[name]))
            with open('results/'+name+'.mean_transpiled_circuit_depth','w') as f:
                f.write(str(mean_depth_transpiled_dict[name]))
            with open('results/'+name+'.max_transpiled_circuit_depth','w') as f:
                f.write(str(max_depth_transpiled_dict[name]))

        first=True
        for r in amplification_factors:
            name=experimentname+"_r"+str(r)
            if result_dict[name] == None:
                result_dict[name] = read_results(name)
            E_dict[name] = expectationvalue_fun(result_dict[name])
            E_av_dict[name] = np.zeros_like(E_dict[name])
            for j in range(1,num_experiments+1):
                E_av_dict[name][j-1] = sum(E_dict[name][0:j])/j
            if first:
                E_av=E_av_dict[name]
                first=False
            else:
                E_av=np.append(E_av,E_av_dict[name])## this is not very efficient coding

    else:
        raise ValueError("not yet implemented, coming soon")

    R=Richardson_extrapolate(E_av.reshape(len(amplification_factors),num_experiments),\
                             np.array(amplification_factors))


    return R, E_dict, E_av_dict,\
           max_depth_dict,mean_depth_dict,\
           max_depth_transpiled_dict,mean_depth_transpiled_dict,\
           experimentname


