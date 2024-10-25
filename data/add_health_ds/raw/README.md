This file reads the Add Health dataset and processes it into a networkx graph.
The data is retrieved from the Add Health Database, which contains several school networks. 
In each network, a node represents a student in that particular school and an edge exists between
two students if they have nominated one another in the Add Health surveys. The data contains demographic information, 
such as self-reported gender and race. We analyze a subset of schools, in which each school is analyzed as 
a separate network (since there are seldom connections reported across schools) and we use race as the sensitive 
attribute of the students. Many schools are still quite segregated, which is apparent in the racial distribution 
of students (e.g. some schools have ver $70\%$ white students, whereas others are predominantly Black or Latino). 
We denote each analyzed school by `Community X' (shorthand comm in code), where X is the id of the school in the anonymized dataset. 
Each school we analyze contains between $500$ and $2,000$ students.

A note about communities for data analysis: comm 6 has predominantly black students of race code = 2, comm 78 has predominantly Asian students of race code = 4 
(but I had looked at the hispanic minority), comm77 I had looked at the black vs non black student populations, comm 41 and 76 I had look at the 
hispanic minority, commm 92, 268, and 271 have predominantly hispanic studentsof race code = 3, 
other communities are predominantly white of race code = 1. 

Data:
- AddHealth/comm' + community + '.paj are the edge files
- AddHealth/comm' + community + 'pajrace.csv are the race files

Code for reading the datasets into networkx graph objects in Python:
- process_AddHealthdata.py
