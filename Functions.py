# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 08:47:20 2020

@author: BHuang
"""

import numpy as np
import pandas as pd
from datetime import datetime,timedelta
from dateutil.relativedelta import relativedelta
from functools import reduce
import tkinter.messagebox
from cvxopt import matrix
# from cvxopt.blas import dot 
from cvxopt.solvers import qp, options,lp
from xgboost.sklearn import XGBClassifier
from functools import reduce
from datetime import datetime,timedelta
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import tkinter.messagebox
from scipy.stats import norm

def test():
    tkinter.messagebox.showwarning(title='Test', message='Why so serious?')
    
def ReadData(path):
    df=pd.read_excel(path,sheet_name='Adj Close')
    df_re=pd.read_excel(path,sheet_name='Return')
    df['Date']=pd.to_datetime(df['Date'])
    df_re['Date']=pd.to_datetime(df_re['Date'])
    df=df.set_index('Date').sort_index(ascending=True).dropna()
    df_re=df_re.set_index('Date').sort_index(ascending=True).dropna()
    return df,df_re


def GeoMean(df,para):
    # name=df.columns
    [timeN,assetN]=df.shape
    GeoMean=[]
    for i in range(assetN):
        a=np.array(df.iloc[:,i]+1)
        pord=np.prod(a)
        b=float(-1+pord**(1/len(a)))
        GeoMean.append(b)
    GeoMean=np.array(GeoMean)
    if para==0:
        return GeoMean
    elif para==1:
        return 12*float(np.mean(GeoMean))

def ArithMean(df):
    [timeN,assetN]=df.shape
    ArithMean=[]
    for i in range(assetN):
        ArithMean.append(float(df.iloc[:,i].mean()*12))
    ArithMean=np.array(ArithMean)
    return float(np.mean(ArithMean))

def sd(w,cov):
    # return np.sqrt(dot(w,Return))
    return float(np.sqrt(reduce(np.dot, [w, cov, w.T])))

def AssetClassCov(df1,df2,df3,df4):
    index=df1.index
    df1=df1.mean(1).to_frame()
    df2=df2.mean(1).to_frame()
    df3=df3.mean(1).to_frame()
    df4=df4.mean(1).to_frame()
    df=pd.concat([df1,df2,df3,df4],axis=1)
    cov=df.cov().values
    return cov

# def sd(w,cov):
#     # return np.sqrt(dot(w,Return))
#     return float(np.sqrt(reduce(np.dot, [w, cov, w.T])))

def XGBoost(returns,factRet):
    [timeN,factorN] = factRet.shape
    [timeN,assetN] = returns.shape
    #Prepare training and preidcting data
    colName=list(factRet.columns)
    f_bar=factRet.tail(2).mean()
    f_bar=pd.DataFrame(f_bar).T
    f_bar.columns=colName    

    factRet=factRet.head(len(factRet)-1)
    xgb = XGBClassifier(learning_rate=0.1,n_estimators=10,
                                max_depth=7,min_child_weight=2,
                                gamma=0.2,subsample=0.8,
                                colsample_bytree=0.6,objective='reg:linear',
                                scale_pos_weight=1,seed=10) 
    mu=[]
    for i in range(assetN):
        xgb.fit(factRet,returns.iloc[:,i])
        mu.append(float(xgb.predict(f_bar)))
    mu=np.array(mu)
    Q = np.array(returns.cov())
    return mu,Q

def AssetClassOptimization(classRe,classCov,lb,ub,given_r_scale):
    #Construct input matrix
    lb=np.array(lb)
    ub=np.array(ub)
    n=len(classRe)
    P=matrix(classCov)
    q=matrix(np.zeros((n, 1)))
    G=np.zeros((2*n,n))
    for i in range(n):
        j=2*i
        G[j,i]=-1
        G[j+1,i]=1
    G=matrix(G)

    h=np.zeros((2*n, 1))
    for i in range(n):
        h[2*i]=-1*lb[i]
        h[2*i+1]=ub[i]
    h=matrix(h)
    aaa=np.ones((1,n))
    A=matrix(np.vstack((aaa,classRe)))

    given_r = []
    risk = []
    weight=[]    
    for temp_r in np.arange(max(min(classRe),0),max(classRe),0.0001):
        try:
            b=matrix(np.array([[1],[temp_r]]))
            # try:
            options['show_progress'] = False
            options['maxiters']=1000
            # options['refinement']=1
            outcome = qp(P,q,G,h,A,b)
            x=np.array(outcome['x'])
            
            
            if outcome['status']!='optimal':
                continue
            given_r.append(temp_r)
            risk.append(sd(x.T,classCov))
            
            weight.append(x.round(4))
        except:
            pass     
    index=int(round(given_r_scale*len(given_r),0))
    # given_r=np.array(given_r)
    # risk=np.array(risk)
    return np.array(weight[index])

def MaxSharpeRatio(Return,Cov,rf,lb,ub):
    #Construct input matrix
    lb=np.array(lb)
    ub=np.array(ub)
    n=len(Return)
    P=matrix(Cov)
    q=matrix(np.zeros((n, 1)))
    G=np.zeros((2*n,n))
    for i in range(n):
        j=2*i
        G[j,i]=-1
        G[j+1,i]=1
    G=matrix(G)

    h=np.zeros((2*n, 1))
    for i in range(n):
        h[2*i]=-1*lb[i]
        h[2*i+1]=ub[i]
    h=matrix(h)
    aaa=np.ones((1,n))
    A=matrix(np.vstack((aaa,Return)))
    
    given_r = []
    risk = []
    weight=[]    
    
    #Make given return increase from minimum possible return to maximum possible return with step length=0.0001
    #And find minimum variance to each given return
    for temp_r in np.arange(max(min(Return),0),max(Return),0.0001):
        try:
            b=matrix(np.array([[1],[temp_r]]))
            # try:
            options['show_progress'] = False
            options['maxiters']=1000
            # options['refinement']=1
            outcome = qp(P,q,G,h,A,b)
            x=np.array(outcome['x'])
            
            
            if outcome['status']!='optimal':
                continue
            given_r.append(temp_r)
            risk.append(sd(x.T,Cov))
            
            weight.append(x.round(4))
        except:
            pass        
    SharpeRatio=(np.array(given_r)-rf)/np.array(risk)
    return weight[SharpeRatio.argmax()]
        
def CVaR_Optimization(df,rf,para):
    inf=np.inf
    alpha=0.95
    Return=GeoMean(df,para)
    given_r=[]
    weight=[]
    CVaR=[]
    for temp_r in np.arange(max(0,min(Return)),max(Return),0.0001):
        timeNum,assetNum=df.shape
        lb=np.concatenate((0*np.ones([1,assetNum]),np.zeros([1,timeNum]),0*np.ones([1,1])),axis=1).T
        bound=[]
        for i in range(len(lb)):
            temp_bound=(float(lb[i]),None)
            bound.append(temp_bound)
        
        aaa=np.concatenate((-df.values,-np.eye(timeNum),-np.ones([timeNum,1])),axis=1)
        bbb=np.concatenate((np.reshape(-Return,[1,len(Return)]),np.zeros([1,timeNum]),np.zeros([1,1])),axis=1)
        A = np.concatenate((aaa,bbb),axis=0)
        
        b=np.concatenate((np.zeros([timeNum,1]),temp_r*np.ones([1,1])),axis=0)
        
        Aeq=np.concatenate((np.ones([1,assetNum]),np.zeros([1,timeNum+1])),axis=1)
        beq=np.ones([1,1])
        
        k=1/((1-alpha)*timeNum)
        c=np.concatenate((np.zeros([assetNum,1]),k*np.ones([timeNum,1]),np.ones([1,1])),axis=0)
        
        outcomes=linprog(c, A_ub=A, b_ub=b, A_eq=Aeq, b_eq=beq,bounds=bound)
        if outcomes.success==True:
            weight.append(np.round(outcomes.x[0:assetNum],4))
            given_r.append(temp_r)
            CVaR.append(round(float(outcomes.fun),4))
    if len(np.unique(CVaR))==1:
        return weight[0]
    else:
        return weight[np.argmin(CVaR)]

def risk_var(asset_return, weight):

# input 1
# format: np.array
# size: N * 1
# weight of each asset in our portfolio



# input 2
# format: dataframe
# table of each equity's return on time basis
# size : N * T

    asset_ticker = list(asset_return.columns)
    
    n=asset_return.shape[1]
    composition = weight

    def simulate_returns(historical_returns, forecast_days):
        return historical_returns.sample(n = forecast_days,
               replace = True).reset_index(drop = True)
    
    # simulate m days in future, we take historical return, select m values with replacement randomly and uniformly
    # resample like bootstrap
    # simulate each asset and multiply by its weight
    
    def simulate_portfolio(historical_returns, composition, forecast_days):
      result = 0
      for t in range(n):
        try:
            weight = composition[t]
        except:
            weight=composition
        name = asset_ticker[t]
        s = simulate_returns(historical_returns[name], 
          forecast_days)
        result = result + s * weight
      return(result)
    
    VaR=[]
    for i in range(100):
        simu_return = simulate_portfolio(asset_return,composition,6)
        sort_return = simu_return.sort_values(ascending=True)
        sort_return.reset_index(drop = "true")
        
        # 99% confident
        confidence_level=0.99
        threshold=int(len(asset_return)*(1-confidence_level))
        var = sort_return.iloc[threshold]
        VaR.append(var)
    return (round(100*np.mean(VaR),2))


def risk_var_norm(mu, Q, weight):
    n=len(weight)
    result = 0
    critical=norm.ppf(0.99)
    for t in range(n):
        result = result + weight[t] * (mu[t] + critical*float(Q[t,t]))
    return result
    
    
    
def snro_gfc_equity(w_e,value_p,equity_drop):
    """ 
    seniro test under global financial crisis case
    """
#equity stress test  
    gfc_e=[]
    for i in range(len(w_e)):
        gfc_e.append(float(w_e[i] * equity_drop.iloc[i]))
    gfc_e_drop = pd.DataFrame(gfc_e).iloc[:,0].sum() * value_p
    return float(gfc_e_drop)

def snro_gfc_fi(w_f,value_f):
#fixed income stress test
    gfc_f_drop = (w_f[0] * (0.048268) + w_f[1] * (-0.1522268)) * value_f

    return float(gfc_f_drop)

def snro_gfc_pe(w_pe,value_pe):
#private equity 
    gfc_pe_drop = w_pe * value_pe * (-0.4638)
    return float(gfc_pe_drop)

def snro_gfc_real(w_r,value_r):
# real estate
    gfc_r_drop = (w_r[0] * (-0.53183924) + w_r[1] * (-0.208404905)) * value_r
       
    return float(gfc_r_drop)
    

def snro_blackMonday(w_e,value_p,equity_drop):
    """
    seniro test under Black Monday case. In this case, it only affect equity. 
    So in that day, the equity drop 22.6%
    """
    black_e=[]
    for i in range(len(w_e)):
        black_e.append(float(w_e[i] * equity_drop.iloc[i]))
    black_e_drop = pd.DataFrame(black_e).iloc[:,0].sum() * value_p
    
    return float(black_e_drop)

def snro_ir_equity(equity,equityPr,w,spx):
    """
    seniro test under 1994 when the interest rate suddenly increase by 1% for no reason.
    """
    # equity stress test  
    ir=[]
    for i in range(equity.shape[1]):
        df=pd.concat([equity.iloc[:,i],spx],axis=1)
        beta = df.cov().values[0,1] / np.var(spx.values)
        ir.append( w[i] * beta * (-0.07) * equityPr[i])
    return float(sum(ir))
    # fixed income stree test
    # 缺数据
def snro_ir_fi(fix,fixPr,w,spx):
    ir=[]
    for i in range(fix.shape[1]):
        df=pd.concat([fix.iloc[:,i],spx],axis=1)
        beta = df.cov().values[0,1] / np.var(spx.values)
        ir.append( w[i] * beta * (-0.07) * fixPr[i])
    return float(sum(ir))
        
# real estate stree test
def snro_ir_real(realestate,realPr,w,spx):
    ir=[]
    for i in range(realestate.shape[1]):
        df=pd.concat([realestate.iloc[:,i],spx],axis=1)
        beta = df.cov().values[0,1] / np.var(spx.values)
        ir.append( w[i] * beta * (-0.07) * realPr[i])
    return float(sum(ir))

def snro_ir_pe(pe,pePr,spx):
    df=pd.concat([pe,spx],axis=1)
    beta = df.cov().values[0,1] / np.var(spx.values)
    ir=beta * (-0.07) * pePr
    return float(ir)
