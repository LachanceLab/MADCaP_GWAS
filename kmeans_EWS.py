"""
This script shows the robustness of the cluster separation between the genetic architectures of Prostate Cancer in the East, West and South regions of the continent. It does so using the genetic variance coefficients (GVC) of the 90 key SNPs estimated in the MADCaP study. GVC estimate for each SNP is calculated in each region using
gvc=2*beta*p*(1-p); where beta is the effect size estimate and p is the allele frequency
To show the robustness of the cluster separation to the standard error of these estimates that can arise due to low allele frequencies and the size of the dataset, effect sizes were sampled from a normal distribution N(Beta,Std_Err) whose mean is the effect size estimate for that SNP in the regional GWAS and Standard Deviation is the Standard error estimated. Sampling performed 10,000 times resulted in 10,000 sampled gvc estimates for each of the East, West and South regions. Kmeans clustering was performed on these vectors. Finally, ANOSIM was used to evaluate if the distance matrix showed significant differences in the three regions.
"""

#!/usr/bin/env python3
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import math
import statistics
import scipy.stats as stats
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
from skbio import DistanceMatrix
from skbio.stats.distance import anosim



statsfile=pd.read_csv("summstats_90", sep=",")

# Regional Effect size estimates for 90 SNPs
e_b=list(statsfile["E_Beta"].fillna(0))
w_b=list(statsfile["W_Beta"].fillna(0))
s_b=list(statsfile["S_Beta"].fillna(0))

# Regional Allele Frequencies for 90 SNPs
e_f=list(statsfile["E_Freq"].fillna(0))
w_f=list(statsfile["W_Freq"].fillna(0))
s_f=list(statsfile["S_Freq"].fillna(0))


# Standard error for regional effect size estimates
e_b_stderr=list(statsfile["E_BetaSE"].fillna(0))
w_b_stderr=list(statsfile["W_BetaSE"].fillna(0))
s_b_stderr=list(statsfile["S_BetaSE"].fillna(0))


# Sample betas from estimate and stderr for each of the 90 snps
n_samples=10000
rand_betas_e=[list(np.random.normal(loc=i,scale=j,size=n_samples)) if j>0 else [0]*n_samples for i,j in zip(e_b,e_b_stderr)]
rand_betas_w=[list(np.random.normal(loc=i,scale=j,size=n_samples)) if j>0 else [0]*n_samples for i,j in zip(w_b,w_b_stderr)]
rand_betas_s=[list(np.random.normal(loc=i,scale=j,size=n_samples)) if j>0 else [0]*n_samples for i,j in zip(s_b,s_b_stderr)]

# Calculate gvc for each sampled effect size with the population allele frequency for each region
randb_gvc_e=[[2*i*i*e_f[snp]*(1-e_f[snp]) for i in rand_betas_e[snp]] for snp in range(len(rand_betas_e))]
randb_gvc_w=[[2*i*i*w_f[snp]*(1-w_f[snp]) for i in rand_betas_w[snp]] for snp in range(len(rand_betas_w))]
randb_gvc_s=[[2*i*i*s_f[snp]*(1-s_f[snp]) for i in rand_betas_s[snp]] for snp in range(len(rand_betas_s))]
randb_gvcT_e=np.array(randb_gvc_e).T
randb_gvcT_w=np.array(randb_gvc_w).T
randb_gvcT_s=np.array(randb_gvc_s).T

matrices = np.concatenate((randb_gvcT_e, randb_gvcT_w, randb_gvcT_s), axis=0)

# Perform K-Means clustering on the combined array
kmeans = KMeans(n_clusters=3, random_state=0).fit(matrices)

# Get the cluster labels for each sample
labels = kmeans.labels_


# Get the cluster centers
cluster_centers = kmeans.cluster_centers_

colors=["green"] * n_samples + ["blue"] * n_samples + ["orange"] * n_samples

pca = PCA(n_components=2)
reduced_matrices = pca.fit_transform(matrices)
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111)
ax.scatter(reduced_matrices[:, 0], reduced_matrices[:, 1], c=colors)
plt.savefig("2Dseparation_EWS_gvc90_10k.pdf",bbox_inches='tight',pad_inches=0.2)


###Cluster separation stats
# Calculate the pairwise distance between centroids
between_cluster_distance = cdist(cluster_centers, cluster_centers)

# Replace the diagonal elements with NaN so they are ignored
np.fill_diagonal(between_cluster_distance, np.nan)

# Get the minimum distance between each pair of clusters
min_distance = np.nanmin(between_cluster_distance, axis=0)

# Calculate the average distance within each cluster
within_cluster_distance = np.zeros(3)
avg_within_cluster_distance = np.zeros(3)

# Calculate the within-cluster distance for each cluster
for i in range(3):
    cluster = matrices[labels == i]
    centroid = cluster_centers[i]
    within_cluster_distance[i] = np.sum(np.linalg.norm(cluster - centroid, axis=1)**2)
    avg_within_cluster_distance[i]=within_cluster_distance[i]/n_samples


cllabels = pd.Categorical(kmeans.labels_)

# Calculate the pairwise distances between points in the matrices
distance_matrix = DistanceMatrix(cdist(matrices, matrices))

# Run ANOSIM on the distance matrix and categorical array
result = anosim(distance_matrix, cllabels, permutations=999)
print(result)
