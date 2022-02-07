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



