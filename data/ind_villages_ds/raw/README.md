We use the communication network of individuals from Banerjee et al (citation below). The dataset contains different demographic attributes for the individual networks and the household networks, 
from which we analyze mothertongue and religion (these are the attributes that have a minority-majority split, while others do not; gender also has this, but I did not analyze gender because
the two genders reported are equal in size in most villages). We note that the individual data contains gender, mothertongue, and religion, while the household data contains only religion, among these. 
We choose the following villages: 
- Mothertongue: villages $5,59,75$. We work with the individual network information.
- Religion: villages $12, 29, 34, 35, 71, 74, 76$. We work with the household network information since the results are quite similar with the individual networks. (Can look at other villages too if interested.)

Citation: Banerjee, Abhijit, Arun G. Chandrasekhar, Esther Duflo, and Matthew O. Jackson. "The diffusion of microfinance." Science 341, no. 6144 (2013): 1236498.

Data: 
- indianvillages/Demographics/household_characteristics.dta and individual_characteristics.dta are the household- and individual-level demographic information.
- indianvillages/Adjacency Matrices/adj_allVillageRelationships_HH_vilno_' + str(vill) + '.csv and indianvillages/Adjacency Matrices/adj_allVillageRelationships_vilno_' + str(vill) + '.csv are the network files for the household-level and the individual-level networks, respectively, for a village vill. 

Code for reading the data into a networkx object in Python: 
- process_IndianVillagesdata.py 
