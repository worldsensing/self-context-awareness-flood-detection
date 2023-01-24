from datetime import datetime

import pandas as pd


def read_csv(file_path, verbose=True):
    if verbose:
        print(f"Reading File from path: {file_path}")
    file = pd.read_csv(file_path)
    if verbose:
        print(file.columns.to_list())
    return file


def show_plot(plt, show=True):
    if show:
        plt.show()
    else:
        plt.clf()
        plt.cla()
        plt.close()


def save_plot(plt, file_path):
    dt_string = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    plt.savefig(f"./output/{file_path}_{dt_string}.png", bbox_inches='tight')
