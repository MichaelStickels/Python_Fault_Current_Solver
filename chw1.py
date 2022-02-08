#----------------------------------------------------------------------------
# Fault Current Solver
#
# UW EE 455 Winter 2022
#
# Created By: Michael Stickels, ...
#
# Created Date: 01/28/2022
#
# Version: 0.1
#
# ---------------------------------------------------------------------------



# Set Parameters and Constants
data_path = "data/CHW1Data.xlsx"


# Imports
import numpy as np
import pandas as pd
import cmath
import math



# Read in excel input file
input = pd.read_excel(data_path, sheet_name=None)
busData = input['BusData'].dropna()
lineData = input['LineData'].dropna()
faultData = input['FaultData'].dropna()



# Calculate parameters
Y_shape = (busData.shape[0], busData.shape[0])
num_busses = Y_shape[0]



a = cmath.rect(1, 120 * math.pi / 180)
a2 = cmath.rect(1, -120 * math.pi / 180)
a_identity = np.matrix([[1, 1, 1],[1, a2, a],[1, a, a2]])



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Build Z Matrices [Z_0], [Z_1], and [Z_2]

# Y_Bus                                                   #### Ignores resistance (unimplemented)
Y_bus = np.zeros(Y_shape, np.complex128)

for a in range(num_busses):

    # y_ii = y_ii + Y_ij
    Y_bus[lineData['From'][a] - 1, lineData['From'][a] - 1] += 1 / (lineData['X, p.u.'][a] * 1j)
    # y_jj = y_jj + Y_ij
    Y_bus[lineData['To'][a] - 1, lineData['To'][a] - 1] += 1 / (lineData['X, p.u.'][a] * 1j)
    # y_ij = y_ij - Y_ij
    Y_bus[lineData['From'][a] - 1, lineData['To'][a] - 1] -= 1 / (lineData['X, p.u.'][a] * 1j)
    # y_ji = y_ji - Y_ij
    Y_bus[lineData['To'][a] - 1, lineData['From'][a] - 1] -= 1 / (lineData['X, p.u.'][a] * 1j)
    # y_ii = y_ii + B/2_ii (shunt admittance)
    Y_bus[lineData['To'][a] - 1, lineData['To'][a] - 1] -= lineData['Bc/2, p.u.'][a] * 1j
    Y_bus[lineData['From'][a] - 1, lineData['From'][a] - 1] -= lineData['Bc/2, p.u.'][a] * 1j
# pd.DataFrame(Y_bus).to_csv("YBusTest.csv")



# Y_g-1
Y_g1 = np.zeros(Y_shape, np.complex128)

for a in range(num_busses):

    # Y_gk-1 = 1 / Z_gk-1
    if(busData['Xg1'][a] != 0):
        Y_g1[a][a] = 1 / (busData['Xg1'][a] * 1j)
# pd.DataFrame(Y_g1).to_csv("Yg1Test.csv")



# Y_D
Y_D = np.zeros(Y_shape, np.complex128)

for a in range(num_busses):

    # Y_Dk = S*_Dk / |V_Fk|^2
    if(busData['Pd p.u.'][a] != 0 or busData['Qd p.u.'][a] != 0):
        Y_D[a][a] = (busData['Pd p.u.'][a] - 1j * busData['Qd p.u.'][a]) / abs(busData['Vf'][a])**2
# pd.DataFrame(Y_D).to_csv("YDTest.csv")



# Y_1 = Y_bus + Y_g-1 + Y_D
Y_1 = Y_bus + Y_g1 + Y_D
# pd.DataFrame(Y_1).to_csv("Y1Test.csv")



# Y_g-2
Y_g2 = np.zeros(Y_shape, np.complex128)

for a in range(num_busses):

    # Y_gk-2 = 1 / Z_gk-2
    if(busData['Xg2'][a] != 0):
        Y_g2[a][a] = 1 / (busData['Xg2'][a] * 1j)
# pd.DataFrame(Y_g2).to_csv("Yg2Test.csv")



# Y_2 = Y_bus + Y_g-2 + Y_D
Y_2 = Y_bus + Y_g2 + Y_D
# pd.DataFrame(Y_2).to_csv("Y2Test.csv")


# Building [Z_0]
#
# Y_bus_0
Y_bus0 = np.zeros(Y_shape, np.complex128)

for a in range(num_busses):

    # y_ii = y_ii + Y_ij
    Y_bus0[lineData['From'][a] - 1, lineData['From'][a] - 1] += 1 / (lineData['X, p.u.'][a] * 1j * 3)
    # y_jj = y_jj + Y_ij
    Y_bus0[lineData['To'][a] - 1, lineData['To'][a] - 1] += 1 / (lineData['X, p.u.'][a] * 1j* 3)
    # y_ij = y_ij - Y_ij
    Y_bus0[lineData['From'][a] - 1, lineData['To'][a] - 1] -= 1 / (lineData['X, p.u.'][a] * 1j* 3)
    # y_ji = y_ji - Y_ij
    Y_bus0[lineData['To'][a] - 1, lineData['From'][a] - 1] -= 1 / (lineData['X, p.u.'][a] * 1j* 3)
    # y_ii = y_ii + B/2_ii (shunt admittance)
    # should this have the * 3 or is the math different?
    Y_bus0[lineData['To'][a] - 1, lineData['To'][a] - 1] -= lineData['Bc/2, p.u.'][a] * 1j* 3
    Y_bus0[lineData['From'][a] - 1, lineData['From'][a] - 1] -= lineData['Bc/2, p.u.'][a] * 1j* 3

    if busData['GenGround'][a] != 0:
        Y_bus0[a,a] += 1 / (busData['Xg0'][a] * 1j)

Y_0 = Y_bus0 + Y_D



# Inverting [Y_s] for [Z_s]
#
# Z_0 (Zero Sequence)
Z_0 = np.linalg.inv(Y_0)
pd.DataFrame(Z_0).to_csv("Z0Test.csv")


# Z_1 (Positive Sequence)
Z_1 = np.linalg.inv(Y_1)
pd.DataFrame(Z_1).to_csv("Z1Test.csv")



# Z_2 (Negative Sequence)
Z_2 = np.linalg.inv(Y_2)
pd.DataFrame(Z_2).to_csv("Z2Test.csv")



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Itereate Faults (from data)

def iterateFaults(faults):

    for a in range(faults.shape[0]):

        type = faults['Type'][a]

        if (type == '3Ph'):

            print('Fault', a + 1)
            print('Type: 3ph')
            print('Bus:', faults['Fault Bus'][a])
            print('Result Bus:', faults['Results Bus'][a])
            print('Z_F:', faults['Zf'][a])
            calculate_3phase(faults['Fault Bus'][a], faults['Results Bus'][a], faults['Zf'][a])
            print()

        elif (type == 'SLG'):

            print('Fault', a + 1)
            print('Type: SLG')
            print('Bus:', faults['Fault Bus'][a])
            print('Result Bus:', faults['Results Bus'][a])
            print('Z_F:', faults['Zf'][a])
            calculate_slg(faults['Fault Bus'][a], faults['Results Bus'][a], faults['Zf'][a])
            print()

        elif (type == 'LL'):

            print('Fault', a + 1)
            print('Type: LL')
            print('Bus:', faults['Fault Bus'][a])
            print('Result Bus:', faults['Results Bus'][a])
            print('Z_F:', faults['Zf'][a])
            calculate_ll(faults['Fault Bus'][a], faults['Results Bus'][a], faults['Zf'][a])
            print()

        elif (type == 'DLG'):

            print('Fault', a + 1)
            print('Type: DLG')
            print('Bus:', faults['Fault Bus'][a])
            print('Result Bus:', faults['Results Bus'][a])
            print('Z_F:', faults['Zf'][a])
            calculate_dlg(faults['Fault Bus'][a], faults['Results Bus'][a], faults['Zf'][a])
            print()

        else:

            print('fault code error')

    return()



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Fault Calculations

# Calculate output for 3phase fault
def calculate_3phase(bus, resultBus, Z_F):

    # I_F = V_F / (Z_nn-1 + Z_F)
    I_F = busData['Vf'][bus - 1] / (Z_1[bus - 1][bus - 1] + Z_F)

    print("I_F = ", abs(I_F))

    return()


# Calculate output for SLG fault
def calculate_slg(bus, resultBus, Z_F):

    # I_F = 3 * V_F / (Z_nn-0 + Z_nn-1 + Z_nn-2 + 3Z_F)
    I_F = 3 * busData['Vf'][bus - 1] / (Z_0[bus - 1][bus - 1] + Z_1[bus - 1][bus - 1] + Z_2[bus - 1][bus - 1] + 3 * Z_F)

    print("I_F = ", abs(I_F))

    return()


# Calculate output for LL fault
def calculate_ll(bus, resultBus, Z_F):

    # I_F = -j * sqrt(3) * V_F / (Z_nn-1 + Z_nn-2 + Z_F)
    I_F = -1j * np.sqrt(3) * busData['Vf'][bus - 1] / (Z_1[bus - 1][bus - 1] + Z_2[bus - 1][bus - 1] + Z_F)

    print("I_F = ", abs(I_F))

    return()


# Calculate output for DLG fault
def calculate_dlg(bus, resultBus, Z_F):

    Z_nn0 = Z_0[bus - 1][bus - 1]
    Z_nn1 = Z_1[bus - 1][bus - 1]
    Z_nn2 = Z_2[bus - 1][bus - 1]
    V_F = busData['Vf'][bus - 1]


    I_n1 = V_F / (Z_nn1 + (Z_nn2 * (Z_nn0 + 3 * Z_F)) / (Z_nn2 + Z_nn0 + 3 * Z_F))
    I_n2 = -I_n1 * (Z_nn0 + 3 * Z_F) / (Z_nn2 + Z_nn0 + 3 * Z_F)
    I_n0 = -I_n1 * (Z_nn2) / (Z_nn2 + Z_nn0 + 3 * Z_F)

    I_s = np.matrix([[I_n0],[I_n1],[I_n2]])
    I_phase = np.matmul(a_identity, I_s)

    print('I_a = ', cmath.polar(I_phase[0]))
    print('I_b = ', cmath.polar(I_phase[1]))
    print('I_c = ', cmath.polar(I_phase[2]))



    return()



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Helper Functions



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Run

print("Fault Current Solver")
print("")

iterateFaults(faultData)

