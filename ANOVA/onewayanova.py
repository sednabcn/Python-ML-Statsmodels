#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 20:33:38 2021

@author: sedna
"""

import pandas as pd
import researchpy as rp

#The function below was created specifically for the one-way ANOVA table results returned for Type II sum of squares

dir="/home/sedna/Donwloads/MATH-TUTOR/UOL/ONE-ANOVA/"
df=pd.read_csv("datos002.csv")
df['Method'].replace({1:"Method A", 2: "Method B",3:"Method C", 4:"Method D"},inplace=True)
print("                 ")

print("                   ")

print("DATASET SCORES BY METHODS:")

print("            ")

dg=pd.DataFrame(index=range(7),columns=['Method A', 'Method B', 'Method C', 'Method D'])
dfe=df.values.reshape(4,7,2)
dfee=dfe[:,:,1]
for j in range(4):
    dg[dg.columns[j]]=dfee[j]
print(dg)


#df['Method'].replace({1:"Method A", 2: "Method B",3:"Method C", 4:"Method D"},inplace=True)
print(rp.summary_cont(df['Scores']))


print("   ")
print("Descritive Statistics for outcome variable DV")

print("   ")
print(rp.summary_cont(df['Scores'].groupby(df['Method'])))
print("     ")
print("ASSUMPTIONS FOR ANOVA TEST")
# INDEPENDENCE
print("       ")
print("INDEPENDENCE")
print("              ")
print("It is Assumed due to the statement ")
print("                 ")
#NORMALITY
print("NORMALITY")
print("             ")
print("Coming from independent t-test framework")
# Coming from independent t-test framework
print("         ")

import pandas as pd
import scipy.stats as stats

print("Kolmogorov-Smirnov")
print("                  ")
print("Method A vs Method B")
ks1=stats.ks_2samp(df['Scores'][df['Method']=='Method A'],
               df['Scores'][df['Method']=='Method B'])
print(ks1)
print("                  ")
print("Method B vs Method C")
ks2=stats.ks_2samp(df['Scores'][df['Method'] == 'Method B'],
               df['Scores'][df['Method'] == 'Method C'])
print(ks2)
print("                  ")
print("Method C vs Method D")
ks3=stats.ks_2samp(df['Scores'][df['Method'] == 'Method C'],
               df['Scores'][df['Method'] == 'Method D'])
print(ks3)
print("         ")

print("Compare more than two samples: Anderson-Darling test for k samples")
print("                  ")
print("Method A vs Method B vs Method C vs Method D")
ks4=stats.anderson_ksamp([df['Scores'][df['Method'] == 'Method A'],
               df['Scores'][df['Method'] == 'Method B'],
               df['Scores'][df['Method'] == 'Method C'],
               df['Scores'][df['Method'] == 'Method D']])
print(ks4)

# Working with residuals
print("  ")
print(" WORKING WITH THE RESIDUALS Scores~C(Method)")
import statsmodels.formula.api as smf
model =smf.ols("Scores~C(Method)", data=df).fit()

print("                  ")
print("Kolmogorov-Smirnov test")
print(stats.kstest(model.resid,'norm'))
print("                  ")
print("Shapiro-Wilk Test")
print("   ")
print(stats.shapiro(model.resid))

print("Figure 1. Probability of model's residuals")
import matplotlib.pyplot as plt

fig = plt.figure(figsize= (10, 10))
ax = fig.add_subplot(111)

normality_plot, stat = stats.probplot(model.resid, plot= plt, rvalue= True)
ax.set_title("Probability plot of model's residuals", fontsize= 20)
ax.set
plt.show()

#Homogeneity Test
print("HOMOGENEITY TEST")
print("    ")
print("LEVENE's TEST")
print("    ")
ks5=stats.levene(df['Scores'][df['Method'] == 'Method A'],
               df['Scores'][df['Method'] == 'Method B'],
               df['Scores'][df['Method'] == 'Method C'],
               df['Scores'][df['Method'] == 'Method D'])
print(ks5)

print("             ")

print(" Figure 2. Box Plot of Scores by method ")

print("                ")

# Checking visually
fig = plt.figure(figsize= (10, 10))
ax = fig.add_subplot(111)

ax.set_title("Box Plot of Scores by method", fontsize= 20)
ax.set

data = [df['Scores'][df['Method'] == 'Method A'],
               df['Scores'][df['Method'] == 'Method B'],
               df['Scores'][df['Method'] == 'Method C'],
               df['Scores'][df['Method'] == 'Method D']]

ax.boxplot(data,
           labels= ['Method A', 'Method B', 'Method C', 'Method D'],
           showmeans= True)

plt.xlabel("Teaching Methods")
plt.ylabel("Scores")

plt.show()

print("     ")
print(" WELCH's T-TEST useful to VARIANCE INEQUALITIES CASES")
print("        ")
# Welch's T-test
print("Method A vs Method B")
print("           ")
ks6=stats.ttest_ind(df['Scores'][df['Method'] == 'Method A'],
               df['Scores'][df['Method'] == 'Method B'], equal_var = False)

print(ks6)
print("Method B vs Method D")
print("                     ")
ks7=stats.ttest_ind(df['Scores'][df['Method'] == 'Method B'],
               df['Scores'][df['Method'] == 'Method D'], equal_var = False)

print(ks7)

print("       ")
print(" ONE-WAY ANOVA TEST")
print("        ")
#ONE-WAY ANOVA WITH PYTHON

ks8=stats.f_oneway(df['Scores'][df['Method'] == 'Method A'],
               df['Scores'][df['Method'] == 'Method B'],
               df['Scores'][df['Method'] == 'Method C'],
               df['Scores'][df['Method'] == 'Method D'])

print(ks8)

import statsmodels.api as sm
from statsmodels.formula.api import ols
print("Scores ~ C(Method)")
print("                     ")
model = ols('Scores ~ C(Method)', data=df).fit()
print("ANOVA TABLE FOR ONE OR MORE FITTED MODELS")
aov_table = sm.stats.anova_lm(model, typ=2)

print(aov_table)



print(" ")
print("EFFECT SIZE IN ANOVA TEST")

aov= aov_table
aov['mean_sq'] = aov[:]['sum_sq']/aov[:]['df']

aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])

aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*aov['mean_sq'][-1]))/(sum(aov['sum_sq'])+aov['mean_sq'][-1])

cols = ['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)', 'eta_sq', 'omega_sq']
aov = aov[cols]

print(aov)

print("                     ")
print(" POST-HOST TESTING ")
# POST-HOC TESTING
print("          ")
# TUKEY HONESTLY SIGNIFFICANCE DIFFERENCE
print("TUKEY HONESTLY SIGNIFFICANCE DIFFERENCE")
#from statsmodels.stats.multicomp import (pairwise_tukeyhsd,
#                                         MultiComparison)
import statsmodels.stats.multicomp as mc

comp = mc.MultiComparison(df['Scores'], df['Method'])
post_hoc_res = comp.tukeyhsd()
post_hoc_res.summary()
print(post_hoc_res)
post_hoc_res.plot_simultaneous(ylabel= "Method", xlabel= "Score Differences")

#BONFERRONI CORRECTION
print("                          ")
print("BONFERRONI CORRECTION")
print("                             ")
import statsmodels.stats.multicomp as mc

comp = mc.MultiComparison(df['Scores'], df['Method'])
tbl, a1, a2 = comp.allpairtest(stats.ttest_ind, method= "bonf")

print(tbl)

#ŠIDÁK CORRECTION (A.K.A. DUNN-ŠIDÁK CORRECTION)
print("                          ")
print("DUNN-SIDAK CORRECTION")
print("                             ")
tbl, a1, a2 = comp.allpairtest(stats.ttest_ind, method= "sidak")

print(tbl)

print("                          ")
print(" ALTERNATIVE METHOD WHEN ASSUMPTIONS IN ANOVA TEST ARE VIOLED")
print("KRUSKAL-WALLIS TEST")
print("                             ")
ks9=stats.kruskal(df['Scores'][df['Method'] == 'Method A'],
               df['Scores'][df['Method'] == 'Method B'],
               df['Scores'][df['Method'] == 'Method C'],
               df['Scores'][df['Method'] == 'Method D'])
print(ks9)
print("                            ")
print(" Figure 3. Multiple Comparisons between All Pairs (Tukey) ")