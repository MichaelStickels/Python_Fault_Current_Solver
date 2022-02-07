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



