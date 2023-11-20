import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import pathlib
import sys


def count_images(directory):
    subdirs = [subdir for subdir in os.listdir(directory) if os.path.isdir(
        os.path.join(directory, subdir))]
    counts = {}
    for subdir in subdirs:
        subdir_path = os.path.join(directory, subdir)
        image_files = [
                f for f in os.listdir(subdir_path)
                if Image.open(os.path.join(subdir_path, f)).format
                in ['JPEG', 'PNG']
        ]
        counts[subdir] = len(image_files)
    return counts


def plot_charts(counts, directory_name):
    labels = list(counts.keys())
    sizes = list(counts.values())
    # Pie chart
    plt.figure(figsize=[10, 10])
    cmap = plt.get_cmap("tab20c")
    colors = cmap(np.arange(len(labels)) % cmap.N)
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors)
    plt.title(f'Pie chart for {directory_name}')
    plt.savefig(f'{directory_name}_pie_chart.png')
    plt.show()
    # Bar chart
    plt.figure(figsize=[10, 10])
    sns.barplot(y=labels, x=sizes, orient='h', palette='viridis')
    plt.title(f'Bar chart for {directory_name}')
    plt.xlabel('Number of Images')
    plt.ylabel('Condition')
    plt.grid(True)
    plt.savefig(f'{directory_name}_bar_chart.png')
    plt.show()
    # Histogram
    plt.figure(figsize=[10, 10])
    colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
    plt.bar(labels, sizes, color=colors)
    plt.title(f'Bar chart for {directory_name}')
    plt.xlabel('Condition')
    plt.ylabel('Number of Images')
    plt.savefig(f'{directory_name}_bar_chart.png')
    plt.show()


if __name__ == '__main__':
    path = pathlib.Path('./data/images/')

    directory = sys.argv[1]
    directory_name = os.path.basename(directory)
    counts = count_images(directory)
    print(counts)
    plot_charts(counts, directory_name)
