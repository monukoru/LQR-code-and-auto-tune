import casadi as ca
import numpy as np

# The raw code for MPC 

#  parameters 
dt = 0.02
N = 20
m = 0.152
g = 9.81



# discrete model matrices
A = np.array([[1, dt],
              [0, 1]])


B = np.array([[0],
              [dt/m]])


# cost matricess
Q = np.diag([20, 5])   # z, vz
R = np.array([[1]])

#  MPC setup 
opti = ca.Opti()

X = opti.variable(2, N+1)
U = opti.variable(1, N)

x0 = opti.parameter(2,1)
xref = opti.parameter(2,1)

cost = 0 # intially it is set to zero 


for k in range(N):
    e = X[:,k] - xref
    cost += ca.mtimes([e.T, Q, e]) + ca.mtimes([U[:,k].T, R, U[:,k]])

    opti.subject_to(X[:,k+1] == A @ X[:,k] + B @ U[:,k])

    # input constraints (tuning)
    opti.subject_to(-5 <= U[:,k])
    opti.subject_to(U[:,k] <= 5)


# terminal cost
eN = X[:,N] - xref
cost += ca.mtimes([eN.T, Q, eN])

opti.subject_to(X[:,0] == x0)
opti.minimize(cost)

# solver
opti.solver('ipopt')

#  simulate 
x = np.array([[32.0],   # initial z
              [0.0]])   # initial vz

xref_val = np.array([[29.22],
                     [0.0]])

for i in range(50):
    opti.set_value(x0, x)
    opti.set_value(xref, xref_val)

    sol = opti.solve()
    u = sol.value(U[:,0])

    # apply control (with gravity compensation)
    thrust = m*g + u[0]

    # simulate next state
    x = A @ x + B * u

    print(f"step {i}: z = {x[0,0]:.2f}, vz = {x[1,0]:.2f}, u = {u[0]:.2f}")

# Nptel IIT Madras theory control system 