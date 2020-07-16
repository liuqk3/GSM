from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas
import os
import numpy as np
import matplotlib.cm as cmx
import matplotlib.colors as colors


def gen_scalar_color_map(ids):
    jet = plt.get_cmap(name='jet')#, lut=len(ids))
    c_norm = colors.Normalize(vmin=min(ids), vmax=max(ids))
    scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=jet)

    return scalar_map

def prepare_data_for_plot(ids, features):

    data = []
    ids_unique = np.unique(ids)
    color_map = gen_scalar_color_map(ids)
    for one_id in ids_unique:
        index = ids == one_id
        color = color_map.to_rgba(one_id)
        one_data = [features[index], color, one_id]
        data.append(one_data)
    return data

def plot_data_in_figure(plot_data, title='', show=False):
    plt.figure()
    for one_plot_data in plot_data:
        one_plot_feat = one_plot_data[0]
        one_color = one_plot_data[1]
        one_id = one_plot_data[2]
        plt.scatter(one_plot_feat[:, 0], one_plot_feat[:, 1], c=one_color, label=str(one_id), s=5, alpha=0.5)
    plt.legend()
    plt.title(title)

    if show:
        plt.show()


if __name__ == '__main__':

    data_path = os.path.join('data', 'MOT17-10-DPM.txt')

    data = pandas.read_csv(data_path).values#[0:100,:]
    ids = data[:, 4].astype(np.int)

    index = ids <= 5
    data = data[index]
    ids = ids[index]

    features_ori = data[:, 6:518]
    features_enh = data[:, 518:1030]

    X_embedded_ori = TSNE(n_components=2).fit_transform(features_ori)
    plot_data = prepare_data_for_plot(ids=ids, features=X_embedded_ori)
    plot_data_in_figure(plot_data, title='origin features')

    X_embedded_enh = TSNE(n_components=2).fit_transform(features_enh)
    plot_data = prepare_data_for_plot(ids=ids, features=X_embedded_enh)
    plot_data_in_figure(plot_data, title='enhanced features')

    plt.show()