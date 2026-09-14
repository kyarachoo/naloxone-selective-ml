import pandas as pd

patients = pd.read_csv('data/synthea/patients.csv')
encounters = pd.read_csv('data/synthea/encounters.csv')
conditions = pd.read_csv('data/synthea/conditions.csv')
medications = pd.read_csv('data/synthea/medications.csv')

print('PATIENTS')
print(patients.shape)
print(patients.columns.tolist())

print('\nENCOUNTERS')
print(encounters.shape)
print(encounters.columns.tolist())

print('\nCONDITIONS')
print(conditions.shape)
print(conditions.columns.tolist())

print('\nMEDICATIONS')
print(medications.shape)
print(medications.columns.tolist())
