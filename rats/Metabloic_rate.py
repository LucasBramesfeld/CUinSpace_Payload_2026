import matplotlib.pyplot as plt
import numpy as np


CMRO2 = 1.97e-6 # mol/g/min - Cerebral metabolic rate of oxygen consumption
RQ = 1.0 # CO2 produced / O2 consumed

mass_of_neurons = 0.05e-3 # g

R = 8.3144626 # m^3 Pa K^-1 mol^-1 - specific gas constant
V = 0.05**3 # m^3 - volume
P = 101325 # Pa - pressure at sea level
T = 37 + 273.15 # kelvin - temperature

intial_percent_O2 = 0.2
intial_percent_CO2 = 0.05

intial_mol = P*V/(R*T)
intial_mol_O2 = P*V/(R*T) * intial_percent_O2
intial_mol_CO2 = P*V/(R*T) * intial_percent_CO2

def O2_consumed(t): # t is in minutes -> mol of O2
    return CMRO2 * mass_of_neurons * t

def percent_O2(t):
    return (intial_mol_O2 - O2_consumed(t))/intial_mol

def percent_CO2(t):
    return (intial_mol_CO2 + O2_consumed(t) * RQ)/intial_mol

x = np.linspace(0, 24*60, 200)

O2 = percent_O2(x)
CO2 = percent_CO2(x)

plt.figure(figsize=(8,5))
plt.plot(x, O2, label="O₂")
plt.plot(x, CO2, label="CO₂")
plt.xlabel("Time (minutes)")
plt.ylabel("Gas concentration")
plt.title("O₂ consumption and CO₂ production in closed volume")
plt.legend()
plt.grid(True)
plt.show()