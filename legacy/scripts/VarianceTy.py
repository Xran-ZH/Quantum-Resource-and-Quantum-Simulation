from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, random_clifford,Pauli,Operator,partial_trace,entropy
import numpy as np
from quantum_simulation_recipe.spin import Nearest_Neighbour_1d
from quantum_simulation_recipe.trotter import *
from quantum_simulation_recipe.bounds import norm, tight_bound, commutator
from quantum_simulation_recipe.plot_config import *

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

#random_brick_block circuit
def random_Brick_frame(n):
    a = random_clifford(2)
    id = Operator.from_label('I')
    if(n%2==0):
        b = id
        i = 2
        while i < n:
            a = a.expand(random_clifford(2))
            b = b.expand(random_clifford(2))
            i += 2
        b = b.expand(id)
    else:
        i = 3
        b = id.expand(random_clifford(2))
        while i < n:
            a = a.expand(random_clifford(2))
            b = b.expand(random_clifford(2))
        a = a.expand(id)
    a = a.to_operator()
    b = b.to_operator()
    return a @ b

#calculate m-part entanglement entropy of a pure state
def partEntropy(states,m):
    entro = []
    for i,state in enumerate(states):
        st = partial_trace(state,list(range(m)))
        entro.append(entropy(st,2))
    return entro

erc = 1e-16
#construct hamiltonian:1-d Ising model with perodic boundary condition
n=10 #num of qubits
d=np.power(2,n)
Jx=1
hx=0.8090
hy=0.9045
Ising = Nearest_Neighbour_1d(n=n,Jx=Jx,hx=hx,hy=hy,pbc=True)

#evolution time \delta t
delta_t = 0.01

#unitary error
exact_U = expH(Ising.ham,delta_t)
trotter_U = pf(h_list=Ising.ham_par,t=delta_t,r=1,order=1)
err_U = exact_U - trotter_U
print(np.real(np.sqrt(np.trace(np.dot(err_U,err_U.conj().T))/d)))

depth = 40
#state with different entanglement
states = []
a = Statevector.from_int(0,d)
states.append(a)
m = 0
while m < depth:#each state evovle a small step of U_0
    a = a.evolve(expH(Ising.ham,m+1))
    states.append(a)
    print(m)
    m += 1
np.save("./vardata/entstates",arr=states)
p2=partEntropy(states,2)
p3=partEntropy(states,3)
p4=partEntropy(states,4)
np.save("./vardata/part2typical",arr=p2)
np.save("./vardata/part3typical",arr=p3)
np.save("./vardata/part4typical",arr=p4)
#trotter error of the evoluted state
count = 2000
moments = []
for j,st0 in enumerate(states):
    if j>1:
        errors = []
        i = 0
        while i < count:
            st = st0.evolve(local_random_clifford(n))
            st = st.evolve(err_U)
            errors.append(np.sqrt(np.real(st.inner(st))))
            print(i)
            i += 1
        np.save(f"./vardata/ent_{j}",arr= errors)