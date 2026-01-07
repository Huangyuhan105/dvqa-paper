import numpy as np
from qiskit import QuantumCircuit, Aer, transpile, execute, BasicAer
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Pauli, SparsePauliOp
from scipy.optimize import minimize
from itertools import product
import time
from NADAM import ADAM
from qiskit.primitives import Estimator
from qiskit_algorithms.gradients import ParamShiftEstimatorGradient
from qiskit.circuit import ParameterVector
### save parameter
import pickle 
import quimb.tensor as qtn
from quimb.tensor import MatrixProductState
import copy
class DVQE:
    def __init__(self, num_qubits, num_subsystems, rank,weights,layer,divided_pauli_terms,bond_dim):
        """
        Initialize the DVQE algorithm.
        :param num_qubits: Total number of qubits in the system.
        :param num_subsystems: Number of subsystems to divide the problem into.
        :param rank: Rank of the decomposition (2 for binary systems).
        """
        self.num_qubits = num_qubits
        self.num_subsystems = num_subsystems
        self.rank = rank
        self.layer=layer
        self.subsystem_qubits = num_qubits // num_subsystems  # Qubits per subsystem
        self.params = [ParameterVector(f'θ_{i}', self.subsystem_qubits*(self.layer+1)) for i in range(num_subsystems)]
        self.weights=weights
        self.pauli_terms=divided_pauli_terms
        self.backend = Aer.get_backend('aer_simulator')  # Backend for simulation
        self.cache_object = {}  # Cache for storing subsystem measurement results
        self.cache_grad = {}
        self.maxiter=200
        self.lr=0.1
        self.optimizer=ADAM(maxiter=self.maxiter, lr=self.lr) # Define the estimator
        self.estimator = Estimator() # Define the gradient
        self.gradient = ParamShiftEstimatorGradient(self.estimator)
        self.bond_dim=bond_dim
        self.C_mps=qtn.MPS_rand_state(L=num_subsystems, bond_dim=self.bond_dim, phys_dim=self.rank, normalize=True, cyclic=False)
        self.num_C=2*self.rank*self.bond_dim+(num_subsystems-2)*(self.bond_dim**2)*self.rank
        self.loss_history = []  # To record loss history for each run

    def _generate_rankary_states(self):
        """
        alpha=(alpha_0,..alphak)
        alpha_0={0,..rank-1}
        Generate all rankary states of length num_subsystems.
        Returns:
        - List of rankary tuples, each representing a state.
        """
        return list(product(range(self.rank), repeat=self.num_subsystems))
    


    def _rankary_to_index(self, rankary_vector):
        """
        Convert a vector of integers to an index based on the given rank.
        :param rankary_vector: List of integers (each element should be in range(rank)).
        :return: Corresponding integer index.
        """
        index = 0
        for bit in rankary_vector:
            index = index * self.rank + bit  
        return index

    def int_to_binary_state(self,alphaj):
        """
        Convert an integer to its corresponding quantum state in computational basis.
        :param alpha: Integer representing the state.
        :param num_qubits: Number of qubits.
        :return: List representing the quantum state in computational basis.
        """
        num_qubits=self.subsystem_qubits
        # Convert alpha to binary and pad with zeros to match num_qubits
        binary_representation = [int(bit) for bit in format(alphaj, f'0{num_qubits}b')]
        return binary_representation
    
    def sub_cir(self,alpha,beta,subsystem_index,real_part):
        num_qubits = self.subsystem_qubits
        qc = QuantumCircuit(num_qubits + 1)  # Add one ancillary qubit

        # Encode |α⟩ on the subsystem qubits
        for i, bit in enumerate(alpha):
            if bit == 1:
                qc.x(i + 1)

        # Prepare initial state: (|0> + |1>) / sqrt(2) on ancilla
        qc.h(0)
        if not real_part:
            qc.s(0)

        # Apply controlled unitary CU
        for i, (a_bit, b_bit) in enumerate(zip(alpha, beta)):
            if a_bit != b_bit:
                qc.cx(0, i + 1)

        # Apply parameterized gates
        for l in range(self.layer):
            for i in range(num_qubits):
                qc.ry(self.params[subsystem_index][i + l * num_qubits], i + 1)
            for i in range(1, num_qubits):
                qc.cx(i, i + 1)
        for i in range(num_qubits):
                qc.ry(self.params[subsystem_index][i + self.layer * num_qubits], i + 1)

        # Apply Hadamard and phase gates for measurement preparation
        qc.h(0)
        if not real_part:
            qc.s(0)
        return qc
    

    def sub_circuit(self, subsystem_index, alpha, beta, paulistring, real_part, params):
        """
        Create the quantum circuit for a single subsystem and compute the expectation value.
        If the result is already cached, reuse it.
        """
        # Create a unique key for the cache based on the inputs
        cache_key = (subsystem_index, tuple(alpha), tuple(beta), paulistring, real_part, tuple(params))

        # Check if the result is already cached
        if cache_key in self.cache_object:
            return self.cache_object[cache_key]
      
        # Check if the Pauli string is all 'I'
 
        if all(p == 'I' for p in str(paulistring)):
            if list(alpha) == list(beta):
                if real_part:
                    return 1.0
                else:
                    return 0.0
            else:
                # If the Pauli string is identity, expectation value is always 0
                return 0.0
        # If not cached, compute the circuit and save the result
        qc=self.sub_cir(alpha,beta,subsystem_index,real_part)

        # Bind parameters
        # param_dict = {self.params[subsystem_index][i]: params[i] for i in range(num_qubits * self.layer)}
        qc = qc.assign_parameters(params)

        # Measure the expectation value
        pauli_op = SparsePauliOp([Pauli(paulistring + 'Z')], [1.0])
        qc.save_expectation_value(pauli_op, range(len(paulistring) + 1))

        # Execute the circuit
        result = execute(qc, self.backend).result()
        expectation_value = result.data(0)['expectation_value']

        self.cache_object[cache_key] = expectation_value
        
        return expectation_value
    
    
    def compute_objective(self, x0,x1):
        """
        Compute the objective function Tr(Hρ) for the current parameters.
        """
        start_time=time.time()
        total = 0.0
        params = x0
        C = x1
        C = C / C.norm()
        self.C_mps=C
        B= copy.copy(C)
        # Split parameters for each subsystem
        subsystem_params = np.split(params, self.num_subsystems)
        pi=np.zeros(len(self.weights),dtype=complex)
        for i in range(len(self.weights)):

            contribution=np.zeros([self.num_subsystems,self.rank,self.rank], dtype=np.complex128)
            for alpha, beta in product(self._generate_rankary_states(), repeat=2):  #r**2k

                alpha_states = np.array([self.int_to_binary_state(a) for a in alpha])
                beta_states = np.array([self.int_to_binary_state(b) for b in beta])
                # Get α and β indices
                alpha_index = alpha
                # self._rankary_to_index(alpha)
                beta_index = beta
                # self._rankary_to_index(beta)
                # Cab=C[alpha_index] * np.conjugate(C[beta_index])
                non_zero_indexs=[]
                for subsystem_index in range(self.num_subsystems):
                    if self.pauli_terms[i][subsystem_index]!='I'*self.subsystem_qubits:
                        real_part=self.sub_circuit(subsystem_index, alpha_states[subsystem_index], beta_states[subsystem_index],
                                        self.pauli_terms[i][subsystem_index], real_part=True,
                                        params=subsystem_params[subsystem_index])
                        imag_part=self.sub_circuit(subsystem_index, alpha_states[subsystem_index], beta_states[subsystem_index],
                                        self.pauli_terms[i][subsystem_index], real_part=False,
                                        params=subsystem_params[subsystem_index])          
                        contribution[subsystem_index,alpha_index[subsystem_index],beta_index[subsystem_index]]=real_part + 1j * imag_part
                        non_zero_indexs.append(subsystem_index)
            if len(non_zero_indexs) == 1:
                mps_C,mps_B=copy.copy(C),copy.copy(B)
                tensor_A_data = contribution[non_zero_indexs[0]]
                tensor_A = qtn.Tensor(data=tensor_A_data, inds=(f"k{non_zero_indexs[0]}", f"k{non_zero_indexs[0]}'"), tags="A")
                mps_C.tensors[non_zero_indexs[0]].reindex({mps_C.tensors[non_zero_indexs[0]].inds[-1]: f"k{non_zero_indexs[0]}'"},inplace=True)
                pi[i]=mps_C @ mps_B @ tensor_A 
            if len(non_zero_indexs) == 2:
                mps_C,mps_B=copy.copy(C),copy.copy(B)
                tensor_A_data = contribution[non_zero_indexs[0]]
                tensor_A = qtn.Tensor(data=tensor_A_data, inds=(f"k{non_zero_indexs[0]}", f"k{non_zero_indexs[0]}'"), tags="A")
                tensor_B_data = contribution[non_zero_indexs[1]]
                tensor_B = qtn.Tensor(data=tensor_B_data, inds=(f"k{non_zero_indexs[1]}", f"k{non_zero_indexs[1]}'"), tags="B")
                mps_C.tensors[non_zero_indexs[0]].reindex({mps_C.tensors[non_zero_indexs[0]].inds[-1]: f"k{non_zero_indexs[0]}'"},inplace=True)
                mps_C.tensors[non_zero_indexs[1]].reindex({mps_C.tensors[non_zero_indexs[1]].inds[-1]: f"k{non_zero_indexs[1]}'"},inplace=True)
                pi[i]=mps_C @ mps_B @ tensor_A @ tensor_B
            if len(non_zero_indexs) == 0:
                pi[i]=1
        total = np.dot(self.weights,pi)
        self.cache_object.clear()  # clear the cache
        self.cache_grad.clear()  # clear the cache
        print(f"the loss is {total}, and waste time {time.time()-start_time}")
        self.loss_history.append(np.real(total))
        return np.real(total)
   
    
    def optimize(self, initial_params):
        """
        Optimize the parameters of the variational circuit.
        :param initial_params: Initial guess for the variational parameters.
        :param pauli_terms: List of Pauli terms for the Hamiltonian.
        :return: Optimized parameters and minimum objective value.
        """
        # result = minimize(self.compute_objective, initial_params, method='COBYLA',options={'maxiter':self.maxiter})
        # self.optimizer.minimize(self.compute_objective, x0=initial_params)
        result = self.optimizer.minimize(self.compute_objective, jac=self.compute_gradient, x0=initial_params[:-self.num_C],x1=self.C_mps)
        return result.x, result.fun
    
    def optimize_rank(self, max_rank, initial_params):
        """
        Perform rank-wise optimization.
        :param max_rank: Maximum rank to optimize.
        :param initial_params: Initial parameters for the lowest rank.
        :return: Optimized parameters for all ranks and their respective energies.
        """
        total_params = initial_params

        while self.rank  <= max_rank:
            print(f"Optimizing for rank {self.rank }...")

            # Initialize DVQE for the current rank
            self.maxiter=200
            #5000*(self.rank )**2 ##800*(self.rank )**2 
            self.lr=0.1 /self.rank
            self.optimizer=ADAM(maxiter=self.maxiter, lr=self.lr)
            # Optimize for the current rank
            optimized_params, min_value = self.optimize(total_params)


            # # Prepare parameters for the next rank
            next_rank = self.rank + 1
            num_next_c_params = next_rank ** self.num_subsystems-self.rank  ** self.num_subsystems

            # Expand C for the next rank
            next_C = (2 * np.random.rand(num_next_c_params) - 1)*0.000001
            # Concatenate the new parameters
            total_params = np.concatenate([optimized_params, next_C])

            # # Move to the next rank
            self.rank  = next_rank
          
       
        return optimized_params, min_value
        
    def grds(self,subsystem_index, alpha, beta, paulistring, real_part, params):
        # Create a unique key for the cache based on the inputs
        cache_key = (subsystem_index, tuple(alpha), tuple(beta), paulistring, real_part, tuple(params))

        # Check if the result is already cached
        if cache_key in self.cache_grad:
            return self.cache_grad[cache_key]
      
        grad_value = self.gradient.run(
                        self.sub_cir(alpha, beta, subsystem_index, real_part),
                        SparsePauliOp(paulistring+'Z', [1.0]),
                        [params]
                    ).result().gradients[0]
        self.cache_grad[cache_key] = grad_value
        return grad_value
    
    def remps(self,mps_C):

        grad_T=[]
        for i in range(self.num_subsystems):
    
            reduced_tensors = [mps_C[idx] for idx in range(mps_C.nsites) if idx != i]
   
            reduced_mps = qtn.MatrixProductState.from_fill_fn(
                fill_fn=lambda shape: reduced_tensors.pop(0),  #
                L=self.num_subsystems - 1,  
                bond_dim=self.bond_dim,
                phys_dim=self.rank,
                shape="lrp",
                site_ind_id="k{}",
                site_tag_id="I{}",
            )
            grad_T.append(reduced_mps)
        return grad_T
    
    def comps(self,T):
        def custom_fill_fn(shape):
            tensor = T.pop(0)  
            return tensor.data.reshape(shape)  
        new_mps = qtn.MatrixProductState.from_fill_fn(
            fill_fn=custom_fill_fn,
            L=self.num_subsystems,
            bond_dim=self.bond_dim,
            phys_dim=self.rank,
            shape="lrp",  # 
            site_ind_id="k{}",
            site_tag_id="I{}",
        )
        return new_mps

    def copy_mps(self,mps_C):
        tensor_list = list(mps_C.tensors)
        def custom_fill_fn(shape):
            tensor = tensor_list.pop(0)
            return tensor.data.reshape(shape)
      
        L = len(mps_C.tensors)      
        bond_dim = max(t.shape[0] for t in mps_C.tensors)  
        phys_dim = mps_C.tensors[0].shape[-1]            
        
        new_mps = qtn.MatrixProductState.from_fill_fn(
            fill_fn=custom_fill_fn,
            L=L,
            bond_dim=bond_dim,
            phys_dim=phys_dim,
            shape="lrp",  
            site_ind_id="k{}",  
            site_tag_id="I{}",  
        )
        return new_mps
    
    def compute_gradient(self, theta_para,C):
        """
        Compute the gradient of the objective function with respect to both theta and C.
        """
        # Extract parameters
        C=self.C_mps
        params = theta_para
        C = C/C.norm()
        self.C_mps=C
        B = self.copy_mps(C)
        # Split θ parameters for each subsystem
        subsystem_params = np.split(params, self.num_subsystems)
        # Initialize gradients
        grad_theta, grad_C = np.zeros_like(params, dtype=float), np.zeros_like(C, dtype=complex)
        grad_theta_i=np.zeros([len(self.weights),len(params)], dtype=float)
        # Precompute alpha and beta states
        rankary_states = self._generate_rankary_states()
        T=[]
        # Loop over all combinations of alpha and beta
        for i, weight in enumerate(self.weights):  # Loop over weights
            F_real_parts = np.zeros([self.num_subsystems,self.rank,self.rank])  
            F_imag_parts = np.zeros([self.num_subsystems,self.rank,self.rank]) 
            grad_F_real_theta = np.zeros([len(params)//self.num_subsystems,self.num_subsystems,self.rank,self.rank]) 
            grad_F_imag_theta = np.zeros([len(params)//self.num_subsystems,self.num_subsystems,self.rank,self.rank]) 
            for alpha, beta in product(rankary_states, repeat=2):
                alpha_index = alpha
                # self._rankary_to_index(alpha)
                beta_index = beta
                # self._rankary_to_index(beta)
                alpha_states = np.array([self.int_to_binary_state(a) for a in alpha])
                beta_states = np.array([self.int_to_binary_state(b) for b in beta])
                non_zero_indexs=[]
                # Compute F_j' and F_j'' for all subsystems
                for j in range(self.num_subsystems):
                    if self.pauli_terms[i][j]!='I'*self.subsystem_qubits:
                        # Compute F_j' (real part) and F_j'' (imaginary part)
                        real_part = self.sub_circuit(
                            j,
                            alpha_states[j],
                            beta_states[j],
                            self.pauli_terms[i][j],  # Use the i-th weight's Pauli terms
                            real_part=True,
                            params=subsystem_params[j]
                        )
                        imag_part = self.sub_circuit(
                            j,
                            alpha_states[j],
                            beta_states[j],
                            self.pauli_terms[i][j],  # Use the i-th weight's Pauli terms
                            real_part=False,
                            params=subsystem_params[j]
                        )
                        # print('....')
                        # print(real_part)
                        # print(alpha_index)
                        # print(beta_index)
                        # print(self.rank)
                        # print(F_real_parts[j,alpha_index,beta_index])
                        F_real_parts[j,alpha_index[j],beta_index[j]]=real_part
                        F_imag_parts[j,alpha_index[j],beta_index[j]]=imag_part
                        # Compute gradient of F_j' and F_j'' with respect to θ
                        # print(self.grds(j, alpha_states[j], beta_states[j], self.pauli_terms[i][j], True, subsystem_params[j]))
                        # print(grad_F_real_theta[:,j,alpha_index,beta_index])
                        grad_F_real_theta[:,j,alpha_index[j],beta_index[j]] = self.grds(j, alpha_states[j], beta_states[j], self.pauli_terms[i][j], True, subsystem_params[j])
                        grad_F_imag_theta[:,j,alpha_index[j],beta_index[j]] = self.grds(j, alpha_states[j], beta_states[j], self.pauli_terms[i][j], False, subsystem_params[j])
                        non_zero_indexs.append(j)
            for index in range(len(params)//self.num_subsystems):
                if len(non_zero_indexs)==2:
                    mps_C=copy.copy(C)
                    mps_B=copy.copy(B)
                    tensor_A_data = F_real_parts[non_zero_indexs[0]]+1j* F_imag_parts[non_zero_indexs[0]]
                    tensor_A = qtn.Tensor(data=tensor_A_data, inds=(f"k{non_zero_indexs[0]}", f"k{non_zero_indexs[0]}'"), tags="A")
                    tensor_B_data = grad_F_real_theta[index,non_zero_indexs[1],:,:]+1j* grad_F_imag_theta[index,non_zero_indexs[1],:,:]
                    tensor_B = qtn.Tensor(data=tensor_B_data, inds=(f"k{non_zero_indexs[1]}", f"k{non_zero_indexs[1]}'"), tags="B")
                    mps_C.tensors[non_zero_indexs[0]].reindex({mps_C.tensors[non_zero_indexs[0]].inds[-1]: f"k{non_zero_indexs[0]}'"},inplace=True)
                    mps_C.tensors[non_zero_indexs[1]].reindex({mps_C.tensors[non_zero_indexs[1]].inds[-1]: f"k{non_zero_indexs[1]}'"},inplace=True)
                    grad_theta_i[i,non_zero_indexs[1]*len(params)//self.num_subsystems+index]=(mps_C @ mps_B @ tensor_A @ tensor_B).real
                    mps_C=copy.copy(C)
                    mps_B=copy.copy(B)
                    tensor_A_data = F_real_parts[non_zero_indexs[1]]+1j* F_imag_parts[non_zero_indexs[1]]
                    tensor_A = qtn.Tensor(data=tensor_A_data, inds=(f"k{non_zero_indexs[1]}", f"k{non_zero_indexs[1]}'"), tags="A")
                    tensor_B_data = grad_F_real_theta[index,non_zero_indexs[0]]+1j* grad_F_imag_theta[index,non_zero_indexs[0]]
                    tensor_B = qtn.Tensor(data=tensor_B_data, inds=(f"k{non_zero_indexs[0]}", f"k{non_zero_indexs[0]}'"), tags="B")
                    mps_C.tensors[non_zero_indexs[1]].reindex({mps_C.tensors[non_zero_indexs[1]].inds[-1]: f"k{non_zero_indexs[1]}'"},inplace=True)
                    mps_C.tensors[non_zero_indexs[0]].reindex({mps_C.tensors[non_zero_indexs[0]].inds[-1]: f"k{non_zero_indexs[0]}'"},inplace=True)
                    grad_theta_i[i,non_zero_indexs[0]*len(params)//self.num_subsystems+index]=(mps_C @ mps_B @ tensor_A @ tensor_B).real
                if len(non_zero_indexs)==1:
                    mps_C=copy.copy(C)
                    mps_B=copy.copy(B)
                    tensor_A_data = grad_F_real_theta[index,non_zero_indexs[0]]+1j* grad_F_imag_theta[index,non_zero_indexs[0]]
                    tensor_A = qtn.Tensor(data=tensor_A_data, inds=(f"k{non_zero_indexs[0]}", f"k{non_zero_indexs[0]}'"), tags="A")
                    mps_C.tensors[non_zero_indexs[0]].reindex({mps_C.tensors[non_zero_indexs[0]].inds[-1]: f"k{non_zero_indexs[0]}'"},inplace=True)
                    grad_theta_i[i,non_zero_indexs[0]*len(params)//self.num_subsystems+index]=(mps_C @ mps_B @ tensor_A).real
            mps_B=copy.copy(B)
            grad_mps_B=self.remps(mps_B)
            if len(non_zero_indexs)==2:
                mps_C=copy.copy(C)
                tensor_A_data = F_real_parts[non_zero_indexs[0]]+1j* F_imag_parts[non_zero_indexs[0]]
                tensor_A = qtn.Tensor(data=tensor_A_data, inds=(f"k{non_zero_indexs[0]}", f"k{non_zero_indexs[0]}'"), tags="A")
                tensor_B_data = F_real_parts[non_zero_indexs[1]]+1j* F_imag_parts[non_zero_indexs[1]]
                tensor_B = qtn.Tensor(data=tensor_B_data, inds=(f"k{non_zero_indexs[1]}", f"k{non_zero_indexs[1]}'"), tags="B")
                mps_C.tensors[non_zero_indexs[0]].reindex({mps_C.tensors[non_zero_indexs[0]].inds[-1]: f"k{non_zero_indexs[0]}'"},inplace=True)
                mps_C.tensors[non_zero_indexs[1]].reindex({mps_C.tensors[non_zero_indexs[1]].inds[-1]: f"k{non_zero_indexs[1]}'"},inplace=True)
                G_C_alpha_k=[]
                for sub in range(self.num_subsystems):
                    G_C_alpha_k.append(grad_mps_B[sub]@mps_C @ tensor_A @ tensor_B)
                T.append(np.array(G_C_alpha_k))
            if len(non_zero_indexs)==1:
                mps_C=copy.copy(C)
                tensor_A_data = F_real_parts[non_zero_indexs[0]]+1j* F_imag_parts[non_zero_indexs[0]]
                tensor_A = qtn.Tensor(data=tensor_A_data, inds=(f"k{non_zero_indexs[0]}", f"k{non_zero_indexs[0]}'"), tags="A")
                mps_C.tensors[non_zero_indexs[0]].reindex({mps_C.tensors[non_zero_indexs[0]].inds[-1]: f"k{non_zero_indexs[0]}'"},inplace=True)
                G_C_alpha_k=[]
                for sub in range(self.num_subsystems):
                    G_C_alpha_k.append( grad_mps_B[sub] @ mps_C @ tensor_A)
                T.append(np.array(G_C_alpha_k))
            if len(non_zero_indexs)==0:
                mps_C=copy.copy(C)
                G_C_alpha_k=[]
                for sub in range(self.num_subsystems):
                    G_C_alpha_k.append(mps_C @ grad_mps_B[sub])
                T.append(np.array(G_C_alpha_k))
        # f = sum(ai * wi for ai, wi in zip(T, self.weights))
        f = sum(ai * wi for ai, wi in zip(T, self.weights))
        grad_theta = np.einsum('i,ij->j', self.weights, grad_theta_i)
        return grad_theta, self.comps(list(f))
