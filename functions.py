import xarray as xr
import pandas as pd
import os
from sklearn import preprocessing, cluster
import scipy
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cf
from matplotlib import cm
import cartopy.mpl.ticker as cticker
import matplotlib.ticker as mticker


def transforma_em_data_frame(caminho, depth):
    data_array = xr.open_dataset(caminho)
    for var in data_array.variables:
        data_array = data_array.where(data_array[var] != 15.489999771118164)
    try:
        depth = int(depth)
        data_array_sel = data_array.sel(depth=depth, method="nearest")
        data_frame = data_array_sel.to_dataframe()
        data_frame.drop("depth", axis=1, inplace=True)
    except:
        data_frame = data_array.to_dataframe()
    data_array.close()
    data_frame.dropna()

    return data_frame


def colocar_na_escala(df):
    df_scaled = df.copy()

    scaler = preprocessing.MinMaxScaler()
    for column in df.columns:
        values = df[column].values.reshape(-1, 1)
        scaled_values = scaler.fit_transform(values)
        df_scaled[column] = scaled_values

    return df_scaled


def agrupamento(df, n_cluster):
    model = cluster.KMeans(n_clusters=n_cluster,  init='k-means++', max_iter=1000, n_init=50, random_state=0, algorithm = "elkan")
    df.dropna(inplace=True)
    df["cluster"] = model.fit_predict(df)

    closest, distances = scipy.cluster.vq.vq(model.cluster_centers_, df.drop("cluster", axis=1).values)

    df["centroids"] = 0
    for i in closest:
        df.loc[i, "centroids"] = 1

    df.dropna(inplace=True)
    return df


def scientific_map(df,extent,title,save_fig):
    plt.rcParams['font.size'] = 22
    plt.rcParams['axes.linewidth'] = 2

    bati = r"A:\Dados Hycom\batimetrias\gebco_2023_n0.0_s-30.0_w-50.0_e-30.0.nc"
    bathy = xr.open_dataset(bati)
    bathy = bathy.where(bathy.elevation <= 0)
    LON, LAT = np.meshgrid(bathy.lon.values, bathy.lat.values)
    Z = bathy.elevation.values

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    land_50m = cf.NaturalEarthFeature(category='physical',
                                      name='land',
                                      scale='10m',
                                      facecolor='lightgrey',
                                      edgecolor='face',
                                      zorder=-1)
    ax.add_feature(cf.BORDERS.with_scale('10m'), zorder=100)
    ax.add_feature(cf.COASTLINE.with_scale('10m'), zorder=100)
    ax.add_feature(land_50m, zorder=100)
    ax.add_feature(cf.OCEAN, facecolor='black')
    ax.add_feature(cf.STATES.with_scale('10m'), zorder=100)

    clus = np.unique(df['cluster'])
    labels = ['0' + str(int(i) + 1) for i in clus]
    colors = iter(cm.rainbow(np.linspace(0, 1, len(clus))))
    for i in np.arange(len(clus)):
        df_p = df[df['cluster'] == clus[i]]
        x,y = df_p['lon'], df_p['lat']
        ax.scatter(x,y,color = next(colors), s = 350, marker='s',edgecolors = None,zorder = 3, label = labels[i])
    ax.legend(title='Clusters',title_fontsize=30,loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(title)

    levels2 = [-100, -50, -25]
    CS_b = ax.contour(LON, LAT, Z, levels2, colors='white', linewidths=1, linestyles='solid', alpha=1, zorder=50)
    x_range = np.arange(extent[0], extent[1] + 2, 2)
    y_range = np.arange(extent[2], extent[3] + 2, 2)
    ax.set_extent(extent)
    ax.set_xticks(x_range, crs=ccrs.PlateCarree())
    ax.set_xticklabels(x_range, weight='bold', fontsize=15)
    ax.set_yticks(y_range, crs=ccrs.PlateCarree())
    ax.set_yticklabels(y_range, weight='bold', fontsize=15)
    ax.yaxis.tick_left()
    ax.xaxis.tick_top()
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    lon_formatter = cticker.LongitudeFormatter()
    lat_formatter = cticker.LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    gl = ax.gridlines(draw_labels=True, linewidth=2, color='black', alpha=0.5, linestyle='--')
    gl.bottom_labels = False
    gl.right_labels = False
    gl.top_labels = False
    gl.left_labels = False
    step = 2
    grid_x = range(extent[0] - 2, extent[1] + 2, step)
    grid_y = range(extent[2] - 2, extent[3] + 2, step)
    gl.xlocator = mticker.FixedLocator(grid_x)
    gl.ylocator = mticker.FixedLocator(grid_y)
    gl.xformatter = cticker.LongitudeFormatter()
    gl.yformatter = cticker.LatitudeFormatter()

    plt.tight_layout()
    if save_fig is not None:
        plt.savefig(save_fig, dpi=300, transparent=False, bbox_inches='tight')
    plt.show()

def lista_variaveis(caminho):
    ds = xr.open_dataset(caminho)
    variaveis = list(ds.data_vars.keys())
    return variaveis

def seleciona_variaveis(variaveis):
    df_sel = df.drop(variaveis, axis=1)
    return df_sel


if __name__ == "__main__":
    #caminho = r"U:\Lucas\agrupamento_dados\Dados\water_temp_salinity.nc4"
    caminho = r"A:\Dados Hycom\area_2\media_temporal_climatologica1\water_temp.nc4"

    df = transforma_em_data_frame(caminho, 0)

    df_scaled = colocar_na_escala(df)

    df_agrupamento = agrupamento(df_scaled, 8)

    df_agrupamento = df_agrupamento.reset_index()

    scientific_map(df_agrupamento, [-41, -35, -22, -14], "K-MEANS superfície", r"A:\paulo.victor\DADOS\kmeans\abrolhos_kmeans8_T_1days_sem_lat_lon_media_climatologica.png")