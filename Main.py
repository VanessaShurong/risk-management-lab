#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 14:33:50 2020

@author: Bokun
"""
import numpy as np
import pandas as pd
from functools import reduce
import tkinter as tk
import tkinter.messagebox
from cvxopt import matrix
from cvxopt.solvers import qp, options
# from xgboost.sklearn import XGBClassifier
# from functools import reduce
import matplotlib.pyplot as plt

from Functions import ReadData,GeoMean,ArithMean,sd,AssetClassCov,XGBoost,\
    MaxSharpeRatio,AssetClassOptimization,CVaR_Optimization,risk_var,risk_var_norm,snro_gfc_equity,\
        snro_gfc_fi,snro_gfc_pe,snro_gfc_real,snro_blackMonday,snro_ir_equity,\
            snro_ir_fi,snro_ir_real,snro_ir_pe
from tkinter import ttk    

#Import data    
PE,PE_re=ReadData('TRPEI.xlsx') #Private Equity
RealEst,RealEst_re=ReadData('Real Estate.xlsx')
FI,FI_re=ReadData('Fixed Income.xlsx')
Equity,Equity_re=ReadData('Equity.xlsx')
MacFactor,MacFactor_re=ReadData('MacroFactor.xlsx')#Macroeconomic Factors

StockFactor=pd.read_excel('StockFactor.xlsx')#Stock factors from F-F 5 factos model
StockFactor['Date']=pd.to_datetime(StockFactor['Date'])
StockFactor=StockFactor.set_index('Date').sort_index(ascending=True).dropna()

SPX,SPXRe=ReadData('SP500.xlsx')#S&P 500 used as Benchmark


RF=pd.read_excel('RiskFreeRate.xlsx')
RF['Date']=pd.to_datetime(RF['Date'])
RF=RF.set_index('Date').sort_index(ascending=True).dropna()

equity_drop = pd.read_excel('ScenarioTest.xlsx', sheet_name = 'GFCreturn',index_col=0).T#Data used for Scenario Test

date=list(Equity.index)
index=list(range(len(date)))
Date2Index=dict(zip(date,index))
Index2Date=dict(zip(index,date))

InitialInvestment=100000

#Investment start date
InvestStart=pd.to_datetime('2015-01-01')
InvestStart=Date2Index[InvestStart]

def forUI1(inputRe,InvestStart,combobox,checkbox):
    checkbox=float(checkbox)
    if combobox=='3m':
        rebalance=3
    elif combobox=='6m':
        rebalance=6
    elif combobox=='12m':
        rebalance=12   
    else:
        tkinter.messagebox.showwarning(title='Warning', message='Invalid Frequency Input' )
#Convert user input into a position between minimum possible return and maximum possible return
#Report errro if input value is too large or too small 
    inputRe_scale=(float(inputRe)-min(Initial_classRe))/(max(Initial_classRe)-min(Initial_classRe))
    if inputRe_scale>1:
        tkinter.messagebox.showwarning(title='Warning', message='Too high! We are going to invest legally, instead of grabing the bank.' )
    elif inputRe_scale<0:
        tkinter.messagebox.showwarning(title='Warning', message='Too low! More trust on us, PLEASE.' )
    else:
        Main(inputRe_scale,0,InvestStart,rebalance,checkbox)

def forUI2(InvestStart,combobox,checkbox):
    checkbox=float(checkbox)
    if combobox=='3m':
        rebalance=3
    elif combobox=='6m':
        rebalance=6
    elif combobox=='12m':
        rebalance=12
    else:
        tkinter.messagebox.showwarning(title='Warning', message='Invalid Frequency Input' )
    Main(0,1,InvestStart,rebalance,checkbox)
        
def Main(inputRe_scale,method,InvestStart,rebalance,checkbox):
    #Method is decided by which button you clicked. 
    #method=0 means invest base on input expected return,
    #method=1 means invest base on maximum sharpe ratio
    RebalancePeriod=rebalance #Rebalance Period is 6 months
    if checkbox==1:
        addmoney=10000
    elif checkbox==0:
        addmoney=0
    global Scn,PortValue,PortWeight,TimeSeries,AnnualRe,VaR,AnnualValue,report
    TrainStart=0
    time0=InvestStart
    
    #Culmulated value for 4 asset classes over whole investment period
    Equity_Value=[]
    FI_Value=[]
    Real_Value=[]
    PE_Value=[]
    
    #VaR for each asset classes
    VaR_equity=[]
    VaR_fi=[]
    VaR_real=[]
    VaR_pe=[]
    
    #Weight for each asset classes
    W1=[]
    W2=[]
    W3=[]
    W4=[]
    
    #Time Series used for plotting
    TimeSeries=[]
    # AnnualRe=[]
    
    #Scenario Test-GFC
    Scn_gfc_equity=[]
    Scn_gfc_fi=[]
    Scn_gfc_real=[]
    Scn_gfc_pe=[]
    
    #Scenario Test - Black Monday
    Scn_blackmonday=[]
    
    #Scenario Test - Interest rate increase
    Scn_ir_equity=[]
    Scn_ir_fi=[]
    Scn_ir_real=[]
    Scn_ir_pe=[]
    
    #Used for calculating portfolio value if we only invest in singal asset or asset class. They are used for comparasion plot
    spx_figure=[]
    equity_figure=[]
    fi_figure=[]
    real_figure=[]
    pe_figure=[]
    Invest_spx=100000
    Invest_equity=100000
    Invest_fi=100000
    Invest_real=100000
    Invest_pe=100000
    
    
    count=1 #Count rebalance times
    TotalInvest=InitialInvestment#Total portfolio value after each rebalance, this variable will be updated everytime after rebalance
    
    while InvestStart<len(date):
        'Initialization: select data between right period.'
        factor=MacFactor[TrainStart:InvestStart+1] 
        stockFactor=0.01*StockFactor[TrainStart:InvestStart+1]
        
        equityRe=Equity_re[TrainStart:InvestStart]  
        equityPrice=Equity[TrainStart:InvestStart]

        fiRe=FI_re[TrainStart:InvestStart]  
        fiPrice=FI[TrainStart:InvestStart]
        
        realRe=RealEst_re[TrainStart:InvestStart]  
        realPrice=RealEst[TrainStart:InvestStart]
        
        peRe=PE_re[TrainStart:InvestStart]  
        pePrice=PE[TrainStart:InvestStart]
        
        spxRe=SPXRe[TrainStart:InvestStart]
        spx=SPX[TrainStart:InvestStart]
    
        rf=float(RF.iloc[InvestStart])*0.01
        
        'Asset Class Optimization'
        #Calculate Geomatric mean for each asset classes
        PE_ExpRe=GeoMean(peRe,1)
        RealEst_ExpRe=GeoMean(realRe,1)
        FI_ExpRe=GeoMean(fiRe,1)
        Equity_ExpRe=GeoMean(equityRe,1)
        
        #return vector and covariance matrix
        classRe=np.array([Equity_ExpRe,FI_ExpRe,RealEst_ExpRe,PE_ExpRe])
        classCov=AssetClassCov(Equity_re,FI_re,RealEst_re,PE_re)    
     
        print('Rebalance #'+str(count))
        #upper and lower bound for asset class optimazation
        lb=0.05*np.ones([4,1])
        ub=0.4*np.ones([4,1])
        
        
        if method==0:
            ClassWeight=AssetClassOptimization(classRe,classCov,lb,ub,inputRe_scale)
        elif method==1:
            ClassWeight=MaxSharpeRatio(classRe,classCov,rf,lb,ub)
        
        #Store optimal weight for each asset class
        W1.append(float(ClassWeight[0]))
        W2.append(float(ClassWeight[1]))
        W3.append(float(ClassWeight[2]))
        W4.append(float(ClassWeight[3]))
        #Actual investment amount for each class
        ClassInvest=ClassWeight*TotalInvest
    
        '--------------------------------Equity----------------------------------'
        EquityInvest=ClassInvest[0]
        AssetNum=equityRe.shape[1]

        #Combine macroeconomic factors and F-F model factors, and it's used for public equity optimization
        CombinedFactor=pd.concat([stockFactor,factor],axis=1)
        #Use XGBoost to find expected return and covariance matrix
        mu,Q=XGBoost(equityRe,CombinedFactor)
        lb=np.zeros([AssetNum,1])
        ub=0.3*np.ones([AssetNum,1])#Investment in each stocks cannot be more than 30%
        
        #if sum of predicted return of each stocks is greater than 0, then invest
        #If not, hold the cash
        if sum(mu)>0:
            #Optimal weight
            weight=MaxSharpeRatio(mu,Q,rf,lb,ub)
            
            #Calculate a VaR, if it's worse than -5%, then use CVaR model to replace the weight before
            temp_var=risk_var(equityRe,weight)
            if temp_var<-5:
                weight=CVaR_Optimization(equityRe,rf,0)
                
            #Store the VaR
            VaR_equity.append(risk_var(equityRe,weight))  
            
            #Calculate the number of stocks we buy base on optimal weight before
            equityPr=Equity.iloc[InvestStart].values
            money=weight*EquityInvest
            amount=np.array([float(money[i])/float(equityPr[i]) for i in range(len(equityPr))])
            
            #This loop is for calculating the value of investment in the following 6 months
            #after each rebalance
            for t_equity in range(RebalancePeriod):
                if InvestStart+t_equity < len(Equity_re):
                    assetPr=Equity.iloc[InvestStart+t_equity].values
                    #Store scenario test
                    Scn_gfc_equity.append(snro_gfc_equity(weight,float(np.dot(amount.T,assetPr)),equity_drop))
                    Scn_blackmonday.append(snro_blackMonday(weight,float(np.dot(amount.T,assetPr)),equity_drop))
                    Scn_ir_equity.append(snro_ir_equity(equityRe,assetPr,weight,spxRe))
                    TimeSeries.append(Index2Date[InvestStart+t_equity])
                    #Store value of this asset class
                    Equity_Value.append(float(np.dot(amount.T,assetPr)))
        else:
            VaR_equity.append(0)  
            for t_equity in range(RebalancePeriod):
                if InvestStart+t_equity < len(Equity_re):
                    Scn_gfc_equity.append(0)
                    Scn_blackmonday.append(0)
                    Scn_ir_equity.append(0)
                    Equity_Value.append(float(EquityInvest))
                    TimeSeries.append(Index2Date[InvestStart+t_equity])
        '--------------------------------Fixed Income------------------------------'
        FIInvest=ClassInvest[1]
        AssetNum=fiRe.shape[1]

        mu,Q=XGBoost(fiRe,factor)
        lb=np.zeros([AssetNum,1])
        ub=np.ones([AssetNum,1])
        if sum(mu)>0:
            weight=MaxSharpeRatio(mu,Q,rf,lb,ub)

            temp_var=risk_var(fiRe,weight)
            if temp_var<-5:
                weight=CVaR_Optimization(fiRe,rf,0)
            VaR_fi.append(risk_var(fiRe,weight))  
            
            fiPr=FI.iloc[InvestStart].values
            money=weight*FIInvest
            amount=np.array([float(money[i])/float(fiPr[i]) for i in range(len(fiPr))])
            
            for t_fi in range(RebalancePeriod):
                if InvestStart+t_fi < len(Equity_re):
                    assetPr=FI.iloc[InvestStart+t_fi].values  
                    Scn_gfc_fi.append(snro_gfc_fi(weight,float(np.dot(amount.T,assetPr))))
                    Scn_ir_fi.append(snro_ir_fi(fiRe,assetPr,weight,spxRe))
                    FI_Value.append(float(np.dot(amount.T,assetPr)))
        else:
            VaR_fi.append(0)
            for t_fi in range(RebalancePeriod):
                if InvestStart+t_fi < len(Equity_re):
                    Scn_gfc_fi.append(0)
                    Scn_ir_fi.append(0)
                    FI_Value.append(float(FIInvest))
        '------------------------------Real Estate---------------------------------'
        RealInvest=ClassInvest[2]
        AssetNum=realRe.shape[1]
        # assetPr=AssetPr.truncate(before=StartDate).truncate(after=EndDate)
        
        # weight_CVaR,given_r,CVaR=CVaR_Optimization(assetRe,0.01)
        mu,Q=XGBoost(realRe,factor)
        lb=np.zeros([AssetNum,1])
        ub=np.ones([AssetNum,1])
        
        if sum(mu)>0:
            weight=MaxSharpeRatio(mu,Q,rf,lb,ub)
            
            temp_var=risk_var(realRe,weight)
            if temp_var<-5:
                weight=CVaR_Optimization(realRe,rf,0)
            VaR_real.append(risk_var(realRe,weight))  
            
            realPr=RealEst.iloc[InvestStart].values
            money=weight*RealInvest
            amount=np.array([float(money[i])/float(realPr[i]) for i in range(len(realPr))])
            
            for t_real in range(RebalancePeriod):
                if InvestStart+t_real < len(Equity_re):
                    assetPr=RealEst.iloc[InvestStart+t_real].values    
                    Scn_gfc_real.append(snro_gfc_real(weight,float(np.dot(amount.T,assetPr))))
                    Scn_ir_real.append(snro_ir_real(realRe,assetPr,weight,spxRe))
                    Real_Value.append(float(np.dot(amount.T,assetPr)))      
        else:
            VaR_real.append(0)  
            for t_real in range(RebalancePeriod):
                if InvestStart+t_real < len(Equity_re):
                    Scn_gfc_real.append(0)
                    Scn_ir_real.append(0)
                    Real_Value.append(float(RealInvest))   
        '---------------------------------PE-------------------------------------'
        #Since there is only one asset in this asset class, things might be easier
        PEInvest=ClassInvest[3]
        AssetNum=peRe.shape[1]
        weight=1
        
        VaR_pe.append(risk_var(peRe,weight))
        pePr=PE.iloc[InvestStart].values
        money=weight*PEInvest
        amount=np.array([float(money)/float(pePr[i]) for i in range(len(pePr))])
        
        for t_pe in range(RebalancePeriod):
            if InvestStart+t_pe < len(Equity_re):
                assetPr=PE.iloc[InvestStart+t_pe].values    
                Scn_gfc_pe.append(snro_gfc_pe(weight,float(np.dot(amount.T,assetPr))))
                Scn_ir_pe.append(snro_ir_pe(peRe,assetPr,spxRe))
                PE_Value.append(float(np.dot(amount.T,assetPr)))     

        '-----------------------------------for Comparasion---------------------------------------'
        #This part is used to generate the value of investment that used for comparsion
        spxPr=SPX.iloc[InvestStart].values
        amount_spx=np.array([float(Invest_spx)/float(spxPr[i]) for i in range(len(spxPr))])    

        Bench_Equity=Equity.iloc[InvestStart].values
        money=np.ones([len(Bench_Equity),1])*Invest_equity/len(Bench_Equity)
        amount_Equity=np.array([float(money[i])/float(Bench_Equity[i]) for i in range(len(Bench_Equity))]) 

        Bench_FI=FI.iloc[InvestStart].values
        money=np.ones([len(Bench_FI),1])*Invest_fi/len(Bench_FI)
        amount_FI=np.array([float(money[i])/float(Bench_FI[i]) for i in range(len(Bench_FI))]) 

        Bench_Real=RealEst.iloc[InvestStart].values
        money=np.ones([len(Bench_Real),1])*Invest_real/len(Bench_Real)
        amount_Real=np.array([float(money[i])/float(Bench_Real[i]) for i in range(len(Bench_Real))]) 
        
        Bench_PE=PE.iloc[InvestStart].values
        amount_PE=np.array([float(Invest_pe)/float(Bench_PE[i]) for i in range(len(Bench_PE))]) 
        
        for t_bench in range(RebalancePeriod):
            if InvestStart+t_bench < len(Equity_re):
                assetPr=SPX.iloc[InvestStart+t_bench].values    
                spx_figure.append(float(np.dot(amount_spx.T,assetPr)))             

                assetPr=Equity.iloc[InvestStart+t_bench].values    
                equity_figure.append(float(np.dot(amount_Equity.T,assetPr)))   

                assetPr=FI.iloc[InvestStart+t_bench].values    
                fi_figure.append(float(np.dot(amount_FI.T,assetPr)))   

                assetPr=RealEst.iloc[InvestStart+t_bench].values    
                real_figure.append(float(np.dot(amount_Real.T,assetPr)))   
                
                assetPr=PE.iloc[InvestStart+t_bench].values    
                pe_figure.append(float(np.dot(amount_PE.T,assetPr)))   
                
        porValue=np.array(Equity_Value)+np.array(FI_Value)+np.array(Real_Value)+np.array(PE_Value)
        # AnnualRe.append(np.log(porValue[-1]/porValue[0]))  

        #Add additional 10000 dollar every 6 months
        if (InvestStart-time0)/6 in np.arange(1,100,1):
            TotalInvest=porValue[-1]+addmoney
            Invest_spx=spx_figure[-1]+addmoney
            Invest_equity=equity_figure[-1]+addmoney
            Invest_fi=fi_figure[-1]+addmoney
            Invest_real=real_figure[-1]+addmoney
            Invest_pe=pe_figure[-1]+addmoney
        else:
            TotalInvest=porValue[-1]
            Invest_spx=spx_figure[-1]
            Invest_equity=equity_figure[-1]
            Invest_fi=fi_figure[-1]
            Invest_real=real_figure[-1]
            Invest_pe=pe_figure[-1]
            
        #Update the investment date and some other floating variable
        TrainStart=TrainStart+RebalancePeriod
        InvestStart=InvestStart+RebalancePeriod
        count=count+1
        
    '---------------------------------------Prepare output-----------------------------------'
    #The model is almost done, we just prepare some output files and dataframes

    #Portfolio Value
    PortValue=np.array(Equity_Value)+np.array(FI_Value)+np.array(Real_Value)+np.array(PE_Value)
    spx_figure=np.array(spx_figure)
    PortValue=pd.DataFrame([PortValue,spx_figure,equity_figure,fi_figure,real_figure,pe_figure]).T
    PortValue.index=TimeSeries
    PortValue.columns=['Portfolio','S&P500','Public Equity','Fixed Income','Real Estate','PE']
    
    #Generate a report about return & volatility for each lines shown in the plot
    report=np.ones([6,2])
    report[0,0]=np.log(PortValue['Portfolio'].values[-1]/PortValue['Portfolio'].values[0])
    report[1,0]=np.log(PortValue['S&P500'].values[-1]/PortValue['Portfolio'].values[0])
    report[2,0]=np.log(PortValue['Public Equity'].values[-1]/PortValue['Portfolio'].values[0])
    report[3,0]=np.log(PortValue['Fixed Income'].values[-1]/PortValue['Portfolio'].values[0])
    report[4,0]=np.log(PortValue['Real Estate'].values[-1]/PortValue['Portfolio'].values[0])
    report[5,0]=np.log(PortValue['PE'].values[-1]/PortValue['Portfolio'].values[0])
    
    report[0,1]=np.std(-1+PortValue['Portfolio'].head(len(PortValue)-1).values/PortValue['Portfolio'].tail(len(PortValue)-1).values)
    report[1,1]=np.std(-1+PortValue['S&P500'].head(len(PortValue)-1).values/PortValue['S&P500'].tail(len(PortValue)-1).values)
    report[2,1]=np.std(-1+PortValue['Public Equity'].head(len(PortValue)-1).values/PortValue['Public Equity'].tail(len(PortValue)-1).values)
    report[3,1]=np.std(-1+PortValue['Fixed Income'].head(len(PortValue)-1).values/PortValue['Fixed Income'].tail(len(PortValue)-1).values)
    report[4,1]=np.std(-1+PortValue['Real Estate'].head(len(PortValue)-1).values/PortValue['Real Estate'].tail(len(PortValue)-1).values)
    report[5,1]= np.std(-1+PortValue['PE'].head(len(PortValue)-1).values/PortValue['PE'].tail(len(PortValue)-1).values)
    
    report=pd.DataFrame(report,index=['Portfolio','S&P500','Public Equity','Fixed Income','Real Estate','PE'],columns=['Total Return','Volatility'])
    report.to_excel('Details for the Portfolio Wealth Plot.xlsx')
    #Portfolio value Plot
    fig1=plt.figure()
    for name in list(PortValue.columns):
        PortValue[name].plot(label=name)
    plt.title('Portfolio Wealth')
    plt.legend()
    fig1.show()
    
    #Scenario test
    Scn=pd.DataFrame([np.array(Scn_gfc_equity),np.array(Scn_gfc_fi),np.array(Scn_gfc_real),np.array(Scn_gfc_pe),np.array(Scn_blackmonday),\
                      np.array(Scn_ir_equity),np.array(Scn_ir_fi),np.array(Scn_ir_real),np.array(Scn_ir_pe)]).T
    Scn.columns=['GFC-Equity','GFC-Fixed Income','GFC-Real Estate','GFC-PE','Black Monday-Equity','Interest Rate-Equity','Interest Rate-FI','Interest Rate-Real Estate','Interest Rate-PE']
    Scn.index=TimeSeries
    Scn.to_excel('Scenario Test for each Asset Class.xlsx')
    
    #VaR
    VaR=pd.DataFrame([VaR_equity,VaR_fi,VaR_real,VaR_pe]).T
    VaR.columns=['VaR-Equity','VaR-FI','VaR-Real Estate','VaR-PE']
    VaR.index=[TimeSeries[i*RebalancePeriod] for i in range(count-1)]
    VaR.to_excel('VaR.xlsx')
    
    #Wight
    PortWeight=pd.DataFrame([W1,W2,W3,W4]).T
    PortWeight.columns=['Equity','Fixed Income','Real Estate','PE']
    PortWeight.index=[TimeSeries[i*RebalancePeriod] for i in range(count-1)]
    PortWeight.plot.area(alpha=1,colormap='Pastel1')
    plt.title('Asset Class Weight')

    
    #Calculate average annual return and total return of the portfolio
    AnnualValue=[PortValue['Portfolio'].values[t*RebalancePeriod] for t in range(count-1)]
    AnnualRe=[np.log(AnnualValue[t+1]/AnnualValue[t]) for t in range(len(AnnualValue)-1)]
    AverageRe=1
    for i in range(len(AnnualRe)):
        AverageRe=AverageRe*(AnnualRe[i]+1)
    AverageRe=AverageRe**(2/len(AnnualRe))-1
    TotalRe=np.log(PortValue['Portfolio'].values[-1]/PortValue['Portfolio'].values[0])
    message='Average annual return is '+str(round(100*AverageRe,2))+'% Total return is '+ str(round(100*TotalRe,2))+'%'
    tkinter.messagebox.showinfo(title='Investment Fished', message=message)

'----------------------------Efficient Frontier-------------------------'
#This part is for generating the efficient frontier and informations shown in the window
Start=pd.to_datetime('2015-01-01')
Start=Date2Index[Start]

Initial_factor=MacFactor[0:Start+1]
Initial_equityRe=Equity_re[0:Start]  
Initial_fiRe=FI_re[0:Start]  
Initial_realRe=RealEst_re[0:Start]  
Initial_peRe=PE_re[0:Start]  

'---------------------Asset Class Optimization------------------------------'
PE_ExpRe=ArithMean(Initial_peRe)
RealEst_ExpRe=ArithMean(Initial_realRe)
FI_ExpRe=ArithMean(Initial_fiRe)
Equity_ExpRe=ArithMean(Initial_equityRe)

Initial_classRe=np.array([Equity_ExpRe,FI_ExpRe,RealEst_ExpRe,PE_ExpRe])
Initial_classCov=AssetClassCov(Equity_re,FI_re,RealEst_re,PE_re)    

Equity_Var=Initial_classCov[0,0]**0.5
FI_Var=Initial_classCov[1,1]**0.5
RealEst_Var=Initial_classCov[2,2]**0.5
PE_Var=Initial_classCov[3,3]**0.5

riskfree=float(RF.iloc[InvestStart])
l1=0.05
u1=0.4
l2=0.05
u2=0.4
l3=0.05
u3=0.4
l4=0.05
u4=0.4

lb=[l1,l2,l3,l4]
ub=[u1,u2,u3,u4]
w=[]
r=[]
vol=[]
#Generate the scatter points by random number generator
for i in range(10000):
    w1=np.random.uniform(l1,u1)
    w2=np.random.uniform(l2,u2)
    w3=np.random.uniform(l3,u3)
    w4=1-w1-w2-w3
    while w4<0 or w4>u4:
        w1=np.random.uniform(l1,u1)
        w2=np.random.uniform(l2,u2)
        w3=np.random.uniform(l3,u3)
        w4=1-w1-w2-w3
    w.append(np.array([w1,w2,w3,w4]))
w=np.array(w)
for i in range(10000):
    r.append(np.dot(w[i,:],Initial_classRe))
    vol.append(np.sqrt(reduce(np.dot, [w[i,:], Initial_classCov, w[i,:].T])))

vol=np.array(vol)
r=np.array(r)-riskfree

#Efficient Frontier
lb=np.array(lb)
ub=np.array(ub)
n=len(Initial_classRe)
P=matrix(Initial_classCov)
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
A=matrix(np.vstack((aaa,Initial_classRe)))

given_r = []
risk = []
weight=[]    
for temp_r in np.arange(max(min(Initial_classRe),0),max(Initial_classRe),0.0001):
    b=matrix(np.array([[1],[temp_r]]))
    # try:
    options['show_progress'] = False
    outcome = qp(P,q,G,h,A,b)
    x=np.array(outcome['x'])

    if outcome['status']!='optimal':
        continue
    given_r.append(temp_r)
    risk.append(sd(x.T,Initial_classCov))
    
    weight.append(x.round(4))
    # except:
        # continue        
risk=np.array(risk) 
given_r=np.array(given_r)-riskfree
SharpeRatio=given_r/risk

fig1=plt.figure()
scat = plt.get_cmap("winter")
sc=plt.scatter(vol,r,s=1,c=r/vol,cmap=scat,marker='o')
cb=plt.colorbar(sc)
cb.set_label('Sharpe Ratio')

plt.plot(risk,given_r,c='r',label='Efficient Frontier')
plt.scatter(risk[risk.argmin()],given_r[risk.argmin()],label='Minimum Variance',c='y',s=50)
plt.scatter(risk[SharpeRatio.argmax()],given_r[SharpeRatio.argmax()],label='Maximum Sharpe Ratio',c='r',s=50)
plt.legend(loc=4)

plt.title('Efficient Frontier')
plt.xlabel('Volatility')
plt.ylabel('Excess Return')

fig1.show() 

'----------------------------User Interface----------------------------'
tkMain=tk.Tk()
# tkMain.wm_attributes('-topmost',1)
tkMain.title('Robo Advisor')

tk.Label(tkMain,text='Expected Annual Return').grid(row=0,column=1)
tk.Label(tkMain,text='  Volatility').grid(row=0,column=2)

tk.Label(tkMain,text='Public Equity').grid(row=1,column=0)
tk.Label(tkMain,text=str(round(Equity_ExpRe-0.001,3))).grid(row=1,column=1)
tk.Label(tkMain,text=str(round(Equity_Var,3))).grid(row=1,column=2)

tk.Label(tkMain,text='Fixed Income').grid(row=2,column=0)
tk.Label(tkMain,text=str(round(FI_ExpRe,3))).grid(row=2,column=1)
tk.Label(tkMain,text=str(round(FI_Var,3))).grid(row=2,column=2)

tk.Label(tkMain,text='Real Estate').grid(row=3,column=0)
tk.Label(tkMain,text=str(round(RealEst_ExpRe+0.001,3))).grid(row=3,column=1)
tk.Label(tkMain,text=str(round(RealEst_Var,3))).grid(row=3,column=2)

tk.Label(tkMain,text='Private Equity').grid(row=4,column=0)
tk.Label(tkMain,text=str(round(PE_ExpRe,3))).grid(row=4,column=1)
tk.Label(tkMain,text=str(round(PE_Var,3))).grid(row=4,column=2)

tk.Label(tkMain,text=' ').grid(row=5,column=1,columnspan=3)

tk.Label(tkMain,text='Expected Annual Return').grid(row=6,column=0)
ExpectedReturn=tk.Entry(tkMain)
ExpectedReturn.grid(row=6,column=1,columnspan=3)

checkbox=tk.IntVar()
addmoney=tk.Checkbutton(tkMain,text='Allow 10K injections every 6 months',variable=checkbox,onvalue=1,offvalue=0)
addmoney.grid(row=7,column=0,columnspan=3)

tk.Label(tkMain,text='Rebalance Period').grid(row=8,column=0)
combobox = ttk.Combobox(tkMain, values=["3m", "6m","12m"])
combobox.current(1)
combobox.grid(row=8,column=1)


Button=tk.Button(tkMain,text='Invest base on input',command= lambda:forUI1(ExpectedReturn.get(),InvestStart,combobox.get(),checkbox.get()))
Button.grid(row=9,column=0,columnspan=3)
Button=tk.Button(tkMain,text='Invest base on MSR',command= lambda:forUI2(InvestStart,combobox.get(),checkbox.get()))
Button.grid(row=10,column=0,columnspan=3)

tkMain.mainloop()


















