import pandas as pd

# Prepare metadata with additional columns
metadata = {
    'Column Name': [],
    'Description': [],
    'API Field Name': [],
    'Data Type': [],
    'Importance Level': [],
    'Potential Use Cases': []
}

# Column data from the provided description
columns = [
    'Hospital Service Area', 'Hospital County', 'Operating Certificate Number', 
    'Permanent Facility Id', 'Facility Name', 'Age Group', 'Zip Code - 3 digits', 
    'Gender', 'Race', 'Ethnicity', 'Length of Stay', 'Type of Admission', 
    'Patient Disposition', 'Discharge Year', 'CCSR Diagnosis Code', 
    'CCSR Diagnosis Description', 'CCSR Procedure Code', 
    'CCSR Procedure Description', 'APR DRG Code', 'APR DRG Description', 
    'APR MDC Code', 'APR MDC Description', 'APR Severity of Illness Code', 
    'APR Severity of Illness Description', 'APR Risk of Mortality', 
    'APR Medical Surgical Description', 'Payment Typology 1', 
    'Payment Typology 2', 'Payment Typology 3', 'Birth Weight', 
    'Emergency Department Indicator', 'Total Charges', 'Total Costs'
]

descriptions = [
    'A description of the Health Service Area (HSA) in which the hospital is located.',
    'A description of the county in which the hospital is located. Blank for records with enhanced de-identification.',
    'The facility Operating Certificate Number as assigned by NYS Department of Health. Blank for records with enhanced de-identification.',
    'Permanent Facility Identifier. Blank for records with enhanced de-identification.',
    'The name of the facility where services were performed based on the Permanent Facility Identifier (PFI), as maintained by the NYSDOH Division of Health Facility Planning.',
    'Age in years at time of discharge.',
    'The first three digits of the patient\'s zip code.',
    'Patient gender.',
    'Patient race.',
    'Patient ethnicity.',
    'The total number of patient days at an acute level and/or other than acute care level (excluding leave of absence days).',
    'A description of the manner in which the patient was admitted to the health care facility.',
    'The patient\'s destination or status upon discharge.',
    'The year of discharge.',
    'AHRQ Clinical Classification Software Refined (CCSR) Diagnosis Category Code.',
    'AHRQ Clinical Classification Software Refined (CCSR) Diagnosis Category Description.',
    'AHRQ Clinical Classification Software Refined (CCSR) ICD-10 Procedure Category Code.',
    'AHRQ Clinical Classification Software Refined (CCSR) ICD-10 Procedure Category Description.',
    'The All Patients Refined Diagnosis Related Groups (APR-DRG) Classification Code.',
    'The APR-DRG Classification Code Description in Calendar Year 2022, Version 39 of the APR- DRG Grouper.',
    'All Patient Refined Major Diagnostic Category (APR MDC) Code.',
    'All Patient Refined Major Diagnostic Category (APR MDC) Description.',
    'All Patient Refined Severity of Illness (APR SOI).',
    'All Patient Refined Severity of Illness (APR SOI) description.',
    'All Patient Refined Risk of Mortality (APR ROM).',
    'The APR-DRG specific classification of Medical, Surgical or Not Applicable.',
    'A description of the type of payment for this occurrence.',
    'A description of the type of payment for this occurrence.',
    'A description of the type of payment for this occurrence.',
    'The neonate birth weight in grams; rounded to nearest 100 g.',
    'The Emergency Department Indicator is set based on the submitted revenue codes.',
    'Total charges for the discharge.',
    'Total estimated cost for the discharge'
]

api_field_names = [
    'hospital_service_area', 'hospital_county', 'operating_certificate_number',
    'permanent_facility_id', 'facility_name', 'age_group', 'zip_code_3_digits',
    'gender', 'race', 'ethnicity', 'length_of_stay', 'type_of_admission',
    'patient_disposition', 'discharge_year', 'ccsr_diagnosis_code',
    'ccsr_diagnosis_description', 'ccsr_procedure_code',
    'ccsr_procedure_description', 'apr_drg_code', 'apr_drg_description',
    'apr_mdc_code', 'apr_mdc_description', 'apr_severity_of_illness_code',
    'apr_severity_of_illness', 'apr_risk_of_mortality',
    'apr_medical_surgical', 'payment_typology_1',
    'payment_typology_2', 'payment_typology_3', 'birth_weight',
    'emergency_department_indicator', 'total_charges', 'total_costs'
]

# Predefined importance levels and use cases
importance_levels = [
    'High', 'Medium', 'Low', 'High', 'Medium', 'High', 'Low', 
    'Medium', 'Low', 'Low', 'High', 'High', 'Medium', 'Low', 
    'Medium', 'High', 'Medium', 'High', 'High', 'High', 
    'Medium', 'High', 'Medium', 'High', 'Medium', 'Medium', 
    'Low', 'Low', 'Low', 'Low', 'Medium', 'High', 'High'
]

use_cases = [
    'Geographic analysis of healthcare services',
    'Regional healthcare distribution',
    'Facility identification',
    'Facility tracking',
    'Facility-specific analysis',
    'Demographic segmentation',
    'Geographical patient distribution',
    'Demographic analysis',
    'Diversity and inclusion studies',
    'Demographic research',
    'Hospital resource utilization',
    'Treatment efficiency analysis',
    'Patient flow management',
    'Temporal healthcare trends',
    'Medical classification',
    'Diagnosis pattern analysis',
    'Procedure classification',
    'Medical procedure insights',
    'Patient grouping',
    'Detailed medical classification',
    'Medical category analysis',
    'Diagnostic category insights',
    'Illness severity assessment',
    'Clinical complexity analysis',
    'Mortality risk evaluation',
    'Treatment type analysis',
    'Payment method analysis',
    'Insurance coverage insights',
    'Billing complexity',
    'Neonatal health tracking',
    'Emergency service utilization',
    'Financial analysis',
    'Cost management'
]

# Populate the metadata dictionary
metadata['Column Name'] = columns
metadata['Description'] = descriptions
metadata['API Field Name'] = api_field_names
metadata['Data Type'] = ['Text'] * len(columns)
metadata['Importance Level'] = importance_levels
metadata['Potential Use Cases'] = use_cases

# Create DataFrame
df = pd.DataFrame(metadata)

# Save to Excel
df.to_excel('healthcare_metadata.xlsx', index=False)
print("Metadata Excel file created successfully!")