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

print('\nENCOUNTER CLASSES')
print(encounters['ENCOUNTERCLASS'].value_counts())

print('\nPOSSIBLE OPIOID-RELATED CONDITIONS')
opioid_conditions = conditions[
    conditions['DESCRIPTION'].str.contains(
        'opioid|overdose|dependence|poison',
        case=False,
        na=False
    )
]

print(opioid_conditions[['START', 'PATIENT', 'ENCOUNTER', 'CODE', 'DESCRIPTION']])

print('\nPOSSIBLE OPIOID-RELATED MEDICATIONS')
opioid_meds = medications[
    medications['DESCRIPTION'].str.contains(
        'naloxone|buprenorphine|methadone|oxycodone|hydrocodone|morphine|fentanyl|tramadol|codeine',
        case=False,
        na=False
    )
]

print(opioid_meds[['START', 'PATIENT', 'ENCOUNTER', 'CODE', 'DESCRIPTION']])
