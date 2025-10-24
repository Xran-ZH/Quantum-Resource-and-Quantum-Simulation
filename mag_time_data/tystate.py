from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, random_clifford,Pauli,Operator,partial_trace,entropy
import numpy as np
from quantum_simulation_recipe.spin import Nearest_Neighbour_1d
from quantum_simulation_recipe.trotter import *
from quantum_simulation_recipe.bounds import norm, tight_bound, commutator
from quantum_simulation_recipe.plot_config import *

def magic(state,pauligroup):#输入Pauli减少运算次数
    print("Calculating...")
    d = len(state)
    m = []
    for j, paulistr in enumerate(pauligroup):
        a = state.evolve(Pauli(paulistr))
        m1 = state.inner(a)
        m1 = np.sqrt(np.real(np.conj(m1)*m1))
        m.append(m1)
    magica = 1-d*np.average(np.power(m,4))
    return magica     

def local_random_clifford(n):
    a = random_clifford(1)
    i = 1
    while i < n:
        a = a.expand(random_clifford(1))
        i += 1
    return a

#n-qubit Pauli group(return a string)
def Pauli_group(n):
    String = ['I','X','Y','Z']
    if n == 1:
        return String
    else:
        a = Pauli_group(n-1)
        j = 0
        b = []
        while j < 4:
            c = String[j]
            i = 0
            while i < np.power(4,n-1):
                b.append(c+a[i])
                i += 1
            j += 1
        return b

#calculate m-part entanglement entropy of a pure state
def partEntropy(states,m):
    entro = []
    for i,state in enumerate(states):
        st = partial_trace(state,list(range(m)))
        entro.append(entropy(st,2))
    return entro

n=10
states=[]
sts=np.load("./mag_time_data/TyStates.npy")
for j,st in enumerate(sts):
    s=Statevector(st)
    states.append(s)
magics=np.load("./mag_time_data/Tymagics.npy").tolist()
P=Pauli_group(n)
for j,st in enumerate(states):
    if j==0:
        magics[0]=0
    if j>0 and j < 41:
        m=magic(st,P)
        magics[j]=m
        print(f"magic{j}=",m)
        np.save("./mag_time_data/Tymagics.npy",arr=magics)