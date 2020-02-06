import re #because we need regular expressions
import numpy as np #because we need the random number generator
from qiskit import QuantumCircuit

twirlingPairs = np.load("TwirlingPairs.npz")["data"]  # created in PauliTwirlingPairFinder.py

def create_Paulitwirled_and_noiseamplificatied_circuit(circuit,r,two_error_map,paulitwirling=True,controlledgatename='cx'):
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
    Pauligateset=['id','x','y','z']
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
                indices_cd = twirlingPairs[0][indices_ab[0]][indices_ab[1]]

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
                # 2      6      10     13
                # 3      7      11     13
                # and local indexation
                # 0,0    1,0    2,0    3,0
                # 0,1    1,1    2,1    3,1
                # 0,2    1,2    2,2    3,2
                # 0,3    1,3    2,3    3,3
                # so we make sure to never draw 0 = (0,0)
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
        raise Exception('E and c must have the same dimension.')
    if n<=1:
        raise Exception('the dimension of E and c must be larger than 1.')
    A = np.zeros((n,n))
    b = np.zeros((n,1))
    #must sum to 1
    A[0,:] = 1
    b[0] = 1
    for k in range(1,n):
        A[k,:] = c**k
    x=np.linalg.solve(A,b)
    return np.dot(np.transpose(E),x)
