import numpy as np
import argparse
from matplotlib import pyplot as plt
from os import sys

def graph_from_csv(filename, multiline = False, clean_fn = lambda x: x.strip(),
        independent_convert_fn = lambda x: int(x), dependent_convert_fn = lambda x: int(x),
        xlabel = "", ylabel = "", title = "", xscale = "linear", yscale = "linear", approximate = False, 
        approximate_size = 1, xbase = 10, ybase = 10):
    """
    Generates a graph from a CSV file. Supports multiple sets of the same independent variable.
    
    Created by Hunter Henrichsen, 2020.
    Params:
    filename -- The CSV file to read.
    multiline -- Whether to average (using np.mean()) the datasets that share an independent variable.
    clean_fn -- The function run on each line to clean it before splitting on commas.
    independent_convert_fn -- The function to convert the independent variable to the desired type.
    dependent_convert_fn -- The function to convert the dependent variable to the desired type.
    """
    data = {}

    # Read data.
    with open(filename, 'r') as file:
        header = file.readline()
        columns = clean_fn(header).split(',')
        for i in range(1, len(columns)):
            data[i] = {
                'name': columns[i],
                'data': {},
            }
            while (line = file.readline()):
                split_line = clean_fn(line).split(',')
            for i in range(1, len(split_line)):
                iv = independent_convert_fn(split_line[0])
                if iv not in data[i]['data']:
                    data[i]['data'][iv] = []
                data[i]['data'][iv].append(dependent_convert_fn(split_line[i]))
        file.close()

    # Process data to averaged.
    if not multiline:
        for key in data:
            current_set = data[key]
            for n in data[key]['data']:
                dataset = data[key]['data'][n]
                data[key]['data'][n] = np.mean(dataset)

    # Begin plotting.
    plt.title(title)
    plt.xscale(xscale, basex=xbase)
    plt.yscale(yscale, basey=ybase)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for key in data:
        current_set = data[key]
        lists = sorted(current_set['data'].items())
        x, y = zip(*lists)
        if approximate:
            approx = np.polyfit(np.log(x) if xscale == "log" else x, np.log(y) if yscale == "log" else y, approximate_size)
            print(current_set['name'], "Approximation: (" + str(ybase ** approx[1]) + ")x^" + str(approx[0]))
        plt.plot(x, y, label=current_set['name'])
    plt.legend()
    plt.show()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transforms a CSV file into a matplotlib graph.')
    parser.add_argument('--xscale', action='store', nargs='?', default='linear')
    parser.add_argument('--yscale', action='store', nargs='?', default='linear')
    parser.add_argument('--xbase', action='store', nargs='?', default=10, type=int)
    parser.add_argument('--ybase', action='store', nargs='?', default=10, type=int)
    parser.add_argument('--xlabel', action='store', nargs='?', default='')
    parser.add_argument('--ylabel', action='store', nargs='?', default='')
    parser.add_argument('--title', action='store', nargs='?', default='')
    parser.add_argument('--approximate', action='store_true', default=False)
    parser.add_argument('--approximateSize', action='store', default=1, type=int)
    parser.add_argument('filename', nargs=1, type=str)
    args = parser.parse_args()
    print(args)
    graph_from_csv(args.filename[0], multiline = False, xscale=args.xscale, xbase=args.xbase, yscale=args.yscale, ybase=args.ybase, xlabel=args.xlabel, ylabel=args.ylabel, title = args.title, approximate=args.approximate, approximate_size = args.approximateSize)
