import pandas as pd
import numpy as np

data = np.loadtxt('cho.txt',delimiter='\t') 
#data = np.loadtxt('iyer.txt',delimiter='\t') 
df = pd.DataFrame(data)  
print(df)
