#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 12:24:34 2021

@author: sedna
"""

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.graphics.factorplots import interaction_plot
import matplotlib.pyplot as plt
from scipy import stats
import researchpy as rp
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#The function below was created specifically for the one-way ANOVA table results returned for Type II sum of squares
df=pd.read_csv("datos003.csv")

print("             ")
print(df.describe())
df['Method'].replace({1:"Method A", 2: "Method B",3:"Method C", 4:"Method D"},inplace=True)
df['Gender'].replace({1:"Female",2:"Male"}, inplace=True)

print("             ")
print("DISTRINUTION OF SCORES BY GENDER AND APPLIED TEACHING METHODS")
print("            ")
arrays0=[
 ['Method','Method','Method','Method'],
 ['Method A','Method B', 'Method C', 'Method D'],
    ]
arrays1=[
    ['Gender','Gender'],
    ['Female','Male']
    ]
k3=arrays1[0][0]
k4=arrays1[1]


data={}
for k1,k2 in zip(*arrays0):
              data.update( \
{(k1,k2):{(k3,k4[0]):df['Scores'][df[k1]==k2][df[k3]==k4[0]].values, \
          (k3,k4[1]):df['Scores'][df[k1]==k2][df[k3]==k4[1]].values }})
dfl=pd.DataFrame.from_dict(data)
print(dfl)

print("                     ")
print("Descritive Statistics for outcome variable Scores")

print("   ")
print(rp.summary_cont(df['Scores'].groupby(df['Method'])))
print("     ")
print(rp.summary_cont(df['Scores'].groupby(df['Gender'])))
print("     ")
# Checking visually
fig = plt.figure(figsize= (10, 10))
ax = fig.add_subplot(111)

ax.set_title("Box Plot of Scores by method vs Gender", fontsize= 20)
ax.set

data1 = [df['Scores'][df['Method'] == 'Method A'][df['Gender']=='Female'],
        df['Scores'][df['Method'] == 'Method B'][df['Gender']=='Female'],
        df['Scores'][df['Method'] == 'Method C'][df['Gender']=='Female'],
        df['Scores'][df['Method'] == 'Method D'][df['Gender']=='Female']]


data2 = [df['Scores'][df['Method'] == 'Method A'][df['Gender']=='Male'],
        df['Scores'][df['Method'] == 'Method B'][df['Gender']=='Male'],
        df['Scores'][df['Method'] == 'Method C'][df['Gender']=='Male'],
        df['Scores'][df['Method'] == 'Method D'][df['Gender']=='Male']]
c1='#EFDECD'
Lb=['A','B','C','D']
bp1 = ax.boxplot(data1,labels=Lb,positions=[1,3,5,7], notch=False, widths=0.35,
                 patch_artist=True, boxprops=dict(facecolor=c1),
                 showmeans= True)
c2='#C0E8D5'
bp2 = ax.boxplot(data2, labels=Lb,positions=[2,4,6,8], notch=False, widths=0.35,
                 patch_artist=True, boxprops=dict(facecolor=c2),
                 showmeans= True)

ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Method & Female', 'Method & Male'], loc='upper right')

plt.xlabel("Teaching Methods")
plt.ylabel("Scores")

plt.show()
print("                   ")
print("DATASET SCORES BY METHODS:")

print("            ")

dg=pd.DataFrame(index=['Female','Male'],columns=['Method A', 'Method B', 'Method C', 'Method D'])
dfe=df.groupby(['Gender','Method'])['Scores'].sum()
dg.loc['Female',:]=dfe['Female'].values
dg.loc['Male',:]=dfe['Male'].values

print(dg)

print("                   ")


print("INTERACTION BETWEEN FACTORS:")


fig = interaction_plot(df.Method, df.Gender, df.Scores,
             colors=['red','blue'], markers=['D','^'], ms=10)
plt.show()
print("                          ")
print("                   ")
print(" ASSUMPTIONS TO TWO-WAY ANOVA")
print("                       ")
print("INDEPENDENCE OF SAMPLES")
print("               ")
print("It has been considered on the statement of this exercise:\n the samples have been selected independently and in a random way")
print("                   ")
print("HOMOGENEITY OF VARIANCES EQUALITIES")

print("                       ")
print(" LEVENE'S TEST ")

print("    ")
ks5=stats.levene(
df['Scores'][df['Method'] == 'Method A'][df['Gender']=='Female'],
df['Scores'][df['Method'] == 'Method B'][df['Gender']=='Female'],
df['Scores'][df['Method'] == 'Method C'][df['Gender']=='Female'],
df['Scores'][df['Method'] == 'Method A'][df['Gender']=='Female'],
df['Scores'][df['Method'] == 'Method A'][df['Gender']=='Male'],
df['Scores'][df['Method'] == 'Method A'][df['Gender']=='Male'],
df['Scores'][df['Method'] == 'Method A'][df['Gender']=='Male'],
df['Scores'][df['Method'] == 'Method D'][df['Gender']=='Male'])
print(ks5)
print("           ")
print(" Calculus of TWO-FACTOR ANOVA by hands")
print("            ")
"""
Calculation of Sum of Squares
The calculations of the sum of squares (the variance in the data)
are quite simple using Python. First, we start with getting the sample size (N)
and the degree of freedoms needed. We will use them later to calculate
the mean square. After we have the degree of freedom we continue with
the calculation of the sum of squares.
"""
#Degrees of Freedom
N = len(df.Scores)

methods=df.Method.unique()
gender=df.Gender.unique()

m_sizes=[sum(df['Method']==l) for l in methods]
g_sizes=[sum(df['Gender']==l) for l in gender]

lM=len(methods)
lG=len(gender)

df_a = lM - 1
df_b = lG - 1
df_axb = df_a*df_b
df_w = N-(df_a+1)*(df_b+1)

#Sum of squares
grand_mean= df['Scores'].values.mean()
print("                      ")
print("grand_mean:",grand_mean)
print("                      ")
#Sum of Squares A - Method

ssq_a = sum([m_sizes[l]*(df[df['Method']==methods[l]]['Scores'].mean()-grand_mean)**2 \
             for l in range(len(m_sizes))])

print("                       ")
print(" Sum of Squares-Method Factor:",ssq_a)
print("                        ")
#Sum of Squares B - Gender

ssq_b = sum([g_sizes[l]*(df[df['Gender']==gender[l]]['Scores'].mean()-grand_mean)**2 \
             for l in range(len(g_sizes))])

print("                       ")
print(" Sum of Squares-Gender Factor:",ssq_b)
print("                        ")
#Sum of Squares Total

ssq_t = sum((df['Scores'] - grand_mean)**2)

print("                       ")
print(" Sum of Squares Total:",ssq_t)
print("                        ")

#Sum of Squares Within (error/residual)

Fc = df[df['Gender']=='Female']
Mc = df[df['Gender']=='Male']
Fc_Method_means = [Fc[Fc.Method == d]['Scores'].mean() for d in Fc.Method]
Mc_Method_means = [Mc[Mc.Method == d]['Scores'].mean() for d in Mc.Method]

ssq_w = sum((Mc.Scores - Mc_Method_means)**2) + sum((Fc.Scores - Fc_Method_means)**2)

print("                       ")
print(" Sum of Squares within (error/residual):",ssq_w)
print("                        ")

#Sum of Squares interaction

#Since we have a two-way design we need to calculate the Sum of Squares for the interaction of A and B.

ssq_axb = ssq_t-ssq_a-ssq_b-ssq_w


print("                       ")
print(" Sum of Squares of Interaction Gender-Method:",ssq_axb)
print("                        ")
#Mean Squares
#We continue with the calculation of the mean square for each factor, the interaction of the factors,
#and within.

#Mean Square A

ms_a = ssq_a/df_a

#Mean Square B
ms_b = ssq_b/df_b
#Mean Square AxB
ms_axb = ssq_axb/df_axb
#Mean Square Within/Error/Residual
ms_w = ssq_w/df_w

#F-ratio
#The F-statistic is simply the mean square for each effect and the interaction
#divided by the mean square for within (error/residual). [Fixed Effects A nad B ]
f_a = ms_a/ms_w
f_b = ms_b/ms_w
f_axb = ms_axb/ms_w

#Obtaining p-values
#We can use the scipy.stats method f.sf to check if our obtained F-ratios
#is above the critical value. Doing that we need to use our F-value for
#each effect and interaction as well as the degrees of freedom for them,
#and the degree of freedom within.

p_a = stats.f.sf(f_a, df_a, df_w)
p_b = stats.f.sf(f_b, df_b, df_w)
p_axb = stats.f.sf(f_axb, df_axb, df_w)

#The results are, right now, stored in a lot of variables.
#To obtain a more readable result we can create a DataFrame that will contain our ANOVA table.

print("Two-way anova with interaction by hands calculations")


results = {'sum_sq':[ssq_a, ssq_b, ssq_axb, ssq_w],
           'df':[df_a, df_b, df_axb, df_w],
           'F':[f_a, f_b, f_axb, 'NaN'],
            'PR(>F)':[p_a, p_b, p_axb, 'NaN']}
columns=['sum_sq', 'df', 'F', 'PR(>F)']
aov_table1 = pd.DataFrame(results, columns=columns,
                          index=['Method', 'Gender',
                          'Method:Gender', 'Residual'])

#As a Psychologist, most of the journals we publish in requires to report effect sizes.
#Common software, such as SPSS has eta squared as output. However,
#eta squared is an overestimation of the effect. To get a less biased effect
#size measure we can use omega squared. The following two functions add
#eta squared and omega squared to the above DataFrame that contains the ANOVA table.

def eta_squared(aov):
    aov['eta_sq'] = 'NaN'
    aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])
    return aov
def omega_squared(aov):
    mse = aov['sum_sq'][-1]/aov['df'][-1]
    aov['omega_sq'] = 'NaN'
    aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*mse))/(sum(aov['sum_sq'])+mse)
    return aov
eta_squared(aov_table1)
omega_squared(aov_table1)
print(aov_table1)
print("                                ")
print("Two-way anova with interaction using statsmodels")

formula = 'Scores ~ C(Method) + C(Gender) + C(Method):C(Gender)'
model = ols(formula, df).fit()
aov_table = anova_lm(model, typ=2)

"""
Statsmodels does not calculate effect sizes. The above function will add omega and eta
 squared effect sizes to the ANOVA table.
"""
eta_squared(aov_table)
omega_squared(aov_table)
print(aov_table.round(4))

res1 = model.resid
fig = sm.qqplot(res1,line='s')
plt.show()

print("                                ")
print("Two-way anova without interaction using statsmodels")

formula = 'Scores ~ C(Method) + C(Gender)'
model = ols(formula, df).fit()
aov_table = anova_lm(model, typ=2)

"""
Statsmodels does not calculate effect sizes. The above function will add omega and eta
 squared effect sizes to the ANOVA table.
"""
eta_squared(aov_table)
omega_squared(aov_table)
print(aov_table.round(4))

res0 = model.resid
fig = sm.qqplot(res0,line='s')
plt.show()
print("             ")
print(" POST-HOST TESTING ")
# POST-HOC TESTING
print("          ")

import pingouin as pg

posthocs = pg.pairwise_ttests(dv='Scores',between=['Gender','Method'],
                              interaction=True, data=df).round(2)
cols=posthocs.columns[[0,1,2,3,6,7,9,10,11]]
print(posthocs[cols])

print("        ")
print("Estimated Marginal Mean Differences ")
import math
def estimated_marginal_means(data,groupA,groupB,DV):
            stats = data.groupby([groupA,groupB])[DV].agg(['mean', 'count', 'std'])
            ci95_hi = []
            ci95_lo = []
            for i in stats.index:
                 m, c, s = stats.loc[i]
                 ci95_hi.append(m + 1.96*s/math.sqrt(c))
                 ci95_lo.append(m - 1.96*s/math.sqrt(c))
            stats['ci95_lo'] = ci95_lo
            stats['ci95_hi'] = ci95_hi
            return stats,print(stats)
stats,_=estimated_marginal_means(df,'Gender','Method','Scores')

stats['mean'].plot.bar(color=['red','green','blue','yellow'])
plt.title('Two-Factors Rankings by Pairs')
plt.show()

