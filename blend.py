import pandas as pd

daishu_sub = pd.read_csv('./daishu_submission.csv')
shiji_sub1 = pd.read_csv('./shiji_submission1.csv')
shiji_sub2 = pd.read_csv('./shiji_submission2.csv')
shiji_sub3 = pd.read_csv('./shiji_submission3.csv')
shiji_sub = shiji_sub1.copy()

targets = [col for col in daishu_sub.columns if col!='sig_id']

shiji_sub[targets] = (shiji_sub1[targets].values+shiji_sub2[targets].values+shiji_sub3[targets].values) / 3
daishu_sub[targets] = daishu_sub[targets].values * 0.7 + shiji_sub[targets].values * 0.3

daishu_sub.to_csv('./submission.csv',index=False)
