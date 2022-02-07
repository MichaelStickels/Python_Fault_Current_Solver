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



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Build Z Busses (Z_0, Z_1, Z_2)

# Y_Bus
# Y_bus_real = np.zeros(Y_shape, np.np.float64)
Y_bus_imaginary = np.zeros(Y_shape, np.float64)

for a in range(lineData.shape[0]):

    # y_ii = y_ii + Y_ij
    Y_bus_imaginary[lineData['From'][a] - 1, lineData['From'][a] - 1] += 1 / lineData['X, p.u.'][a]
    # y_jj = y_jj + Y_ij
    Y_bus_imaginary[lineData['To'][a] - 1, lineData['To'][a] - 1] += 1 / lineData['X, p.u.'][a]
    # y_ij = y_ij - Y_ij
    Y_bus_imaginary[lineData['From'][a] - 1, lineData['To'][a] - 1] -= 1 / lineData['X, p.u.'][a]
    # y_ji = y_ji - Y_ij
    Y_bus_imaginary[lineData['To'][a] - 1, lineData['From'][a] - 1] -= 1 / lineData['X, p.u.'][a]
    # y_ii = y_ii + B/2_ii (shunt admittance)
    Y_bus_imaginary[lineData['To'][a] - 1, lineData['To'][a] - 1] -= lineData['Bc/2, p.u.'][a]
    Y_bus_imaginary[lineData['From'][a] - 1, lineData['From'][a] - 1] -= lineData['Bc/2, p.u.'][a]
    

pd.DataFrame(Y_bus_imaginary).to_csv("YBusTest.csv")


print(Y_bus_imaginary)


# Z_0 (Zero Sequence)


# Z_1 (Positive Sequence)


# Z_2 (Negative Sequence)






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



