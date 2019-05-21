import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# This script generates the data analysis plot presented in the paper.

def process_label(lab):
    l, r = lab.split('_')
    return f'{l}-{r}'


def draw_user_plots(means, counts, idx, axxar):
    """
    Draws the particular plots with user data analysis in the given cells of the whole plot.
    :param means: Mean rates for given lay times
    :param counts: Total measure counts for given lay times
    :param idx: The horizontal index for these plots at the whole plot
    :param axxar: The array of axes
    :return: Nothing
    """
    lay_time_keys = means.keys().levels[0].values.tolist()
    lay_time_keys.sort(key=lambda x: int(x.split('_')[0]))

    total_keys = means.keys().levels[1].values.tolist()
    total_keys.sort()

    plot_keys = []  # total sleep time
    mean_plot_values = []  # mean rate
    count_plot_values = []  # measures count
    plot_labels = []  # lay time

    for lt_key in lay_time_keys:
        plot_cur_keys = []
        mean_plot_cur_values = []
        count_plot_cur_values = []
        for t_key in total_keys:
            mean = means.get((lt_key, t_key), default=-1)
            if mean != -1:
                plot_cur_keys.append(t_key)
                mean_plot_cur_values.append(mean)
                count_plot_cur_values.append(counts.get((lt_key, t_key)))

        if len(plot_cur_keys) > 0:
            plot_keys.append(plot_cur_keys)
            mean_plot_values.append(mean_plot_cur_values)
            count_plot_values.append(count_plot_cur_values)
            plot_labels.append(lt_key)

    plot_count = len(plot_labels)

    for i in range(plot_count):
        axxar[0, idx - 1].plot(plot_keys[i], mean_plot_values[i],  label=process_label(plot_labels[i]))
        axxar[0, idx - 1].set_xticks([0, 2, 4, 6, 8, 10, 12])

    legend_handles = [
        plt.Line2D([], [], ls='', marker='o', color=axxar[0, idx - 1].lines[i]._color)
        for i in range(plot_count)
    ]
    legend_labels = [process_label(plot_labels[i]) for i in range(plot_count)]
    axxar[0, idx - 1].legend(
        handles=legend_handles, labels=legend_labels,
        bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.
    )

    for i in range(plot_count):
        axxar[1, idx - 1].plot(plot_keys[i], count_plot_values[i], label=process_label(plot_labels[i]))
        axxar[0, idx - 1].set_xticks([0, 2, 4, 6, 8, 10, 12])

    axxar[1, idx - 1].set_title(f'User #{idx}')


def main():
    ds = pd.read_csv('../data/sleepdata_classified.csv')
    grouped = ds.groupby(['user_id', 'lay_time', 'total']).agg({'rate': [np.size, np.mean]})

    i = 1
    user_ids = ds['user_id'].unique()
    user_count = len(user_ids)

    f, axes_array = plt.subplots(2, user_count, sharex=True, sharey='row', figsize=(11, 4.3))
    for user_id in user_ids:
        means = grouped['rate']['mean'][user_id]
        counts = grouped['rate']['size'][user_id]

        draw_user_plots(means, counts, i, axes_array)

        i += 1

    label_fontsize = 12

    for ax in axes_array[0, :].flat:
        ax.set_xlabel('Sleep duration (hours)', fontsize=label_fontsize)
        ax.set_ylabel('Rating', fontsize=label_fontsize)

    for ax in axes_array[1, :].flat:
        ax.set_xlabel('Sleep duration (hours)', fontsize=label_fontsize)
        ax.set_ylabel('Measures', fontsize=label_fontsize)

    for ax in axes_array.flat:
        ax.label_outer()

    plt.show()


if __name__ == '__main__':
    main()
