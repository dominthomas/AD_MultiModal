import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

pandas2ri.activate()


readRDS = robjects.r['readRDS']
df = readRDS('/home/dthomas/test.rds')
pandas2ri.deactivate()
print(df.shape)
