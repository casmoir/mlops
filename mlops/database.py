from sqlalchemy import create_engine
import pandas as pd



#engine = create_engine(postgresql://username:password@host:5432/mlops)


def create_database(df):
    engine = create_engine('postgresql://postgres:password@localhost:5432/mlops')
    df.to_sql('boston', con=engine, if_exists='replace')




data_url = '/Users/anastasiaraeva/mlops/mlops/mlops/data/boston.csv'
raw_df = pd.DataFrame(pd.read_csv(data_url, sep = ';'))
raw_df = raw_df.iloc[: , 1:]
y = raw_df['target']
x = raw_df.drop('target', axis = 1)


#print(x)

engine = create_database(raw_df)