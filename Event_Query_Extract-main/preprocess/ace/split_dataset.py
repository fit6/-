import pandas as pd
from sklearn import model_selection
# 读入数据
df = pd.read_csv(r'../../data/DuEE1.0/after-ace/duee_train.csv')
X = df[['token', 'offset', 'trigger_offset', 'trigger_type', 'trigger_cluster', 'trigger_arguments']]
X_train, X_test = model_selection.train_test_split(X, test_size = 0.2, random_state = 1234)