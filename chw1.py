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



# Parameters and Constants
data_path = "data/CHW1Data.xlsx"



# Imports
import numpy as np
import pandas as pd



# Read in excel input file
input = pd.read_excel(data_path, sheet_name=None)
busData = input['BusData'].dropna()
lineData = input['LineData'].dropna()
faultData = input['FaultData'].dropna()
# print(busData)
# print(lineData)
# print(faultData)


# Calculate parameters
Y_shape = (busData.shape[0], busData.shape[0])

num_busses = Y_shape[0]



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Build Z Busses (Z_0, Z_1, Z_2)

# Y_Bus                                                     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Calculate directly as complex?
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

pd.DataFrame(Y_bus).to_csv("YBusTest.csv")



# Y_g-1
Y_g1 = np.zeros(Y_shape, np.complex128)

for a in range(num_busses):

    # Y_gk-1 = 1 / Z_gk=1
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



# Z_0 (Zero Sequence)
# Z_0 = np.linalg.inv(Y_0)


# pd.DataFrame(Z_0).to_csv("Z0Test.csv")



# Z_1 (Positive Sequence)
Z_1 = np.linalg.inv(Y_1)

pd.DataFrame(Z_1).to_csv("Z1Test.csv")



# Z_2 (Negative Sequence)
# Z_2 = np.linalg.inv(Y_2)


# pd.DataFrame(Z_2).to_csv("Z2Test.csv")





# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Fault Calculations

# Calculate output for 3phase fault
def calculate_3phase():

    return()


# Calculate output for SLG fault
def calculate_slg():

    return()


# Calculate output for LL fault
def calculate_ll():

    return()


# Calculate output for DLG fault
def calculate_dlg():

    return()



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Helper Functions



