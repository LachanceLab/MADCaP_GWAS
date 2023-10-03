"""
This script shows the difference in variance of two key factors contributing to the difference in the genetic architectures of Prostate Cancer in the East, West and South regions of the continent. It does so by separately quantifying the variance of allele frequency and effect size estimates for each SNP and which of these contributes more to the genetic architectures of Prostate Cancer in different regions of Africa. The genetic architectures are represented using the genetic variance coefficients (GVC) of the 90 key SNPs estimated in the MADCaP study. GVC estimate for each SNP is calculated in each region using
gvc=2*beta*p*(1-p); where beta is the effect size estimate and p is the allele frequency
To show the variance of GVC arising from the variance of these estimates, effect sizes were sampled from a normal distribution N(Beta,Std_Err) whose mean is the effect size estimate for that SNP in the regional GWAS and Standard Deviation is the Standard error from the GWAS. Similarly, allele frequencies were sampled from a normal distribution N(AlleleFrequency,Std_Err). Sampling performed 10,000 times resulted in 10,000 sampled gvc estimates for each of the East, West and South regions from sampled effect sizes and 10,000 from sampled allele frequencies. The Euclidean distance between the GVC vectors of East and West, West and South, and East and South were calculated with both sets of 10,000 GVC vectors to show the variance in gvc arising due to each factor and how they contribute to the differences in genetic architecture.
"""

#!/usr/bin/env python3
import pandas as pd
import numpy as np
import math
import statistics
import scipy.stats as stats
import matplotlib.pyplot as plt

statsfile=pd.read_csv("90_SNPs_polarized_final.txt", sep=",")

n_samples=10000

# Regional Effect size estimates for 90 SNPs
e_b=list(statsfile["E_Beta_Effect"].fillna(0))
w_b=list(statsfile["W_Beta_Effect"].fillna(0))
s_b=list(statsfile["S_Beta_Effect"].fillna(0))

# Regional Allele frequency estimates for 90 SNPs
e_f=list(statsfile["E_FreqTestedAllele"].fillna(0))
w_f=list(statsfile["W_FreqTestedAllele"].fillna(0))
s_f=list(statsfile["S_FreqTestedAllele"].fillna(0))

# Standard error for regional effect size estimates
e_b_stderr=list(statsfile["E_Beta_StdErr"].fillna(0))
w_b_stderr=list(statsfile["W_Beta_StdErr"].fillna(0))
s_b_stderr=list(statsfile["S_Beta_StdErr"].fillna(0))

# Standard error for regional allele frequency estimates
e_f_stderr=list(statsfile["E_AF_StdErr"].fillna(0))
w_f_stderr=list(statsfile["W_AF_StdErr"].fillna(0))
s_f_stderr=list(statsfile["S_AF_StdErr"].fillna(0))

# Sample betas from estimate and stderr for each of the 90 snps
rand_betas_e=[list(np.random.normal(loc=i,scale=j,size=n_samples)) if j>0 else [0]*n_samples for i,j in zip(e_b,e_b_stderr)]
rand_betas_w=[list(np.random.normal(loc=i,scale=j,size=n_samples)) if j>0 else [0]*n_samples for i,j in zip(w_b,w_b_stderr)]
rand_betas_s=[list(np.random.normal(loc=i,scale=j,size=n_samples)) if j>0 else [0]*n_samples for i,j in zip(s_b,s_b_stderr)]

# Calculate gvc for each sampled effect size vector with the population allele frequency for each region
randb_gvc_e=[[2*i*i*e_f[snp]*(1-e_f[snp]) for i in rand_betas_e[snp]] for snp in range(len(rand_betas_e))]
randb_gvc_w=[[2*i*i*w_f[snp]*(1-w_f[snp]) for i in rand_betas_w[snp]] for snp in range(len(rand_betas_w))]
randb_gvc_s=[[2*i*i*s_f[snp]*(1-s_f[snp]) for i in rand_betas_s[snp]] for snp in range(len(rand_betas_s))]

# Sum of gvcs
sum_randb_gvc_e=[sum([randb_gvc_e[i][snpset] for i in range(0,92)]) for snpset in range(0,n_samples)]
sum_randb_gvc_w=[sum([randb_gvc_w[i][snpset] for i in range(0,92)]) for snpset in range(0,n_samples)]
sum_randb_gvc_s=[sum([randb_gvc_s[i][snpset] for i in range(0,92)]) for snpset in range(0,n_samples)]

# Proportion of gvc
randb_propgvc_e=[[randb_gvc_e[i][j]/sum_randb_gvc_e[j] for j in range(0,n_samples)] for i in range(0,92)]
randb_propgvc_w=[[randb_gvc_w[i][j]/sum_randb_gvc_w[j] for j in range(0,n_samples)] for i in range(0,92)]
randb_propgvc_s=[[randb_gvc_s[i][j]/sum_randb_gvc_s[j] for j in range(0,n_samples)] for i in range(0,92)]

# Euclidean distance between gvcs from sampled effect sizes from each pair of regions 
dist_randbfixedf_ew=[math.dist([randb_propgvc_e[i][j] for i in range(len(randb_propgvc_e))],[randb_propgvc_w[i][j] for i in range(len(randb_propgvc_w))]) for j in range(n_samples)]
dist_randbfixedf_sw=[math.dist([randb_propgvc_s[i][j] for i in range(len(randb_propgvc_s))],[randb_propgvc_w[i][j] for i in range(len(randb_propgvc_w))]) for j in range(n_samples)]
dist_randbfixedf_es=[math.dist([randb_propgvc_e[i][j] for i in range(len(randb_propgvc_e))],[randb_propgvc_s[i][j] for i in range(len(randb_propgvc_s))]) for j in range(n_samples)]

# Sample allele frequencies from mean estimate and stderr for each of the 90 snps
rand_freqs_e=[list(np.random.normal(loc=i,scale=j,size=n_samples)) if j>0 else [0]*n_samples for i,j in zip(e_f,e_f_stderr)]
rand_freqs_w=[list(np.random.normal(loc=i,scale=j,size=n_samples)) if j>0 else [0]*n_samples for i,j in zip(w_f,w_f_stderr)]
rand_freqs_s=[list(np.random.normal(loc=i,scale=j,size=n_samples)) if j>0 else [0]*n_samples for i,j in zip(s_f,s_f_stderr)]

# Calculate gvc for each sampled allele frequency  vector with the population allele frequency for each region
randf_gvc_e=[[2*i*(1-i)*e_b[snp]*e_b[snp] for i in rand_freqs_e[snp]] for snp in range(len(rand_freqs_e))]
randf_gvc_w=[[2*i*(1-i)*w_b[snp]*w_b[snp] for i in rand_freqs_w[snp]] for snp in range(len(rand_freqs_w))]
randf_gvc_s=[[2*i*(1-i)*s_b[snp]*s_b[snp] for i in rand_freqs_s[snp]] for snp in range(len(rand_freqs_s))]

# Sum of gvcs
sum_randf_gvc_e=[sum([randf_gvc_e[i][snpset] for i in range(0,92)]) for snpset in range(0,n_samples)]
sum_randf_gvc_w=[sum([randf_gvc_w[i][snpset] for i in range(0,92)]) for snpset in range(0,n_samples)]
sum_randf_gvc_s=[sum([randf_gvc_s[i][snpset] for i in range(0,92)]) for snpset in range(0,n_samples)]

# Proportion of gvc
randf_propgvc_e=[[randf_gvc_e[i][j]/sum_randf_gvc_e[j] for j in range(0,n_samples)] for i in range(0,92)]
randf_propgvc_w=[[randf_gvc_w[i][j]/sum_randf_gvc_w[j] for j in range(0,n_samples)] for i in range(0,92)]
randf_propgvc_s=[[randf_gvc_s[i][j]/sum_randf_gvc_s[j] for j in range(0,n_samples)] for i in range(0,92)]

# Euclidean distance between gvcs from sampled allele frequencies from each pair of regions 
dist_randffixedb_ew=[math.dist([randf_propgvc_e[i][j] for i in range(len(randf_propgvc_e))],[randf_propgvc_w[i][j] for i in range(len(randf_propgvc_w))]) for j in range(n_samples)]
dist_randffixedb_sw=[math.dist([randf_propgvc_s[i][j] for i in range(len(randf_propgvc_s))],[randf_propgvc_w[i][j] for i in range(len(randf_propgvc_w))]) for j in range(n_samples)]
dist_randffixedb_es=[math.dist([randf_propgvc_e[i][j] for i in range(len(randf_propgvc_e))],[randf_propgvc_s[i][j] for i in range(len(randf_propgvc_s))]) for j in range(n_samples)]

maxrange=max(max(dist_randbfixedf_ew),max(dist_randbfixedf_sw),max(dist_randbfixedf_es))
print(maxrange)

# Plot histograms of Euclidean distances between gvcs calculated from sampled effect sizes and sampled allele frequencies between East and West
bins=100
bins = np.histogram(np.hstack((dist_randbfixedf_ew, dist_randffixedb_ew)), bins=bins, range=[0,maxrange])[1]
print(str(statistics.mean(dist_randbfixedf_ew))+","+str(statistics.stdev(dist_randbfixedf_ew)))
print(str(statistics.mean(dist_randffixedb_ew))+","+str(statistics.stdev(dist_randffixedb_ew)))
# print(stats.mannwhitneyu(dist_randbfixedf_ew,dist_randffixedb_ew))
# print(stats.ttest_rel(dist_randbfixedf_ew,dist_randffixedb_ew))

plt.figure(figsize=(7,7))
plt.hist(dist_randbfixedf_ew, bins, range=[0,maxrange], alpha=0.5, label='Sampled effect sizes')
plt.hist(dist_randffixedb_ew, bins, range=[0,maxrange], alpha=0.5, label='Sampled allele frequencies')
plt.legend(loc='upper right')
plt.savefig( "EW_samplebetafreq_separate.pdf",bbox_inches='tight', pad_inches=0.2)


# Plot histograms of Euclidean distances between gvcs calculated from sampled effect sizes and sampled allele frequencies between South and West
bins=100
bins = np.histogram(np.hstack((dist_randbfixedf_sw, dist_randffixedb_sw)), bins=bins, range=[0,maxrange])[1]
print(str(statistics.mean(dist_randbfixedf_sw))+","+str(statistics.stdev(dist_randbfixedf_sw)))
print(str(statistics.mean(dist_randffixedb_sw))+","+str(statistics.stdev(dist_randffixedb_sw)))
# print(stats.ttest_rel(dist_randbfixedf_sw,dist_randffixedb_sw))

plt.figure(figsize=(7,7))
plt.hist(dist_randbfixedf_sw, bins, range=[0,maxrange],alpha=0.5, label='Sampled effect sizes')
plt.hist(dist_randffixedb_sw, bins, range=[0,maxrange],alpha=0.5, label='Sampled allele frequencies')
plt.legend(loc='upper right')
plt.savefig( "SW_samplebetafreq_separate.pdf",bbox_inches='tight', pad_inches=0.2)

# Plot histograms of Euclidean distances between gvcs calculated from sampled effect sizes and sampled allele frequencies between South and East
bins=100
bins = np.histogram(np.hstack((dist_randbfixedf_es, dist_randffixedb_es)), bins=bins, range=[0,maxrange])[1]
print(str(statistics.mean(dist_randbfixedf_es))+","+str(statistics.stdev(dist_randbfixedf_es)))
print(str(statistics.mean(dist_randffixedb_es))+","+str(statistics.stdev(dist_randffixedb_es)))
# print(stats.ttest_rel(dist_randbfixedf_es,dist_randffixedb_es))
plt.figure(figsize=(7,7))
plt.hist(dist_randbfixedf_es, bins, range=[0,maxrange], alpha=0.5, label='Sampled effect sizes')
plt.hist(dist_randffixedb_es, bins, range=[0,maxrange], alpha=0.5, label='Sampled allele frequencies')
plt.legend(loc='upper right')
plt.savefig( "ES_samplebetafreq_separate.pdf",bbox_inches='tight', pad_inches=0.2)
