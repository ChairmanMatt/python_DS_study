import numpy as np
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
# function to visualize data
def visualize(data, n_sample=25):
    fig = plt.figure(figsize=(10, 10))
    num = int(n_sample**0.5)
    gs = gridspec.GridSpec(num, num)
    fig.subplots_adjust(wspace=0.01, hspace=0.02)
    l = len(data.iloc[1, :])
    ll = int(l**0.5)
    for i in range(num):
        for j in range(num):
            temp = np.array(data.loc[(num*i + j), :]).reshape(ll, ll)
            ax = plt.subplot(gs[i, j])
            ax.imshow(temp, cmap=plt.cm.gray, interpolation='nearest')
            ax.axis('off')
    plt.show()
    
    
import pandas as pd
df = pd.read_csv("./data/train.csv")
df_data = df.drop("label", 1)
df_label = df.label

visualize(df_data)


from sklearn.decomposition import PCA
def visualizePCA(data, n_component=100):
    # remain first n components
    pca = PCA(n_components = n_component)
    df_data = pca.fit_transform(data) 
    # convert back to original data
    df_data = pca.inverse_transform(df_data)
    # convert to DataFrame and plot
    df_data = pd.DataFrame(df_data)
    visualize(df_data)
    
visualizePCA(df_data) # takes a small amount of time to run

visualizePCA(df_data, 20)

visualizePCA(df_data, 10)

# PCA is different from feature selection by "combining" features into new features
# (and assuming they are not correlated)
