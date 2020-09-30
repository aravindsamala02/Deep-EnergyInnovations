# Energy Innovations - Machine learning project
Several models were tested to accurately predict the energy demand for facilities at Arizona State University.

## The Dataset
The Original Dataset contains the hourly energy demand (in kW) for all 100 facilities across Arizona State University, Tempe Campus. Additional Information about the Semester months, weekday, holiday and outside weather conditions is acquired and appended to the dataset. 

The Input variables include:
1. Hour of the day (Dummy encoded)
2. Weekday or Weekend (Binary labeled)
3. SemesterOn boolean (Binary labeled)
4. Holiday (Binary labeled) 
5. Average Outside Temperature (Numeric)
6. Energy demand (Numeric in KW) - Output variable

The Building occupancy is indirectly measured from first four variables and various machine learning models starting some simple regression to Nueral Networks are tested. A final Neural Network model predicted with 


