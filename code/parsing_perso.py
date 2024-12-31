# importing required modules
import argparse
import os
import sys

from proc import execute
from check_arguments import argChecker

#?- create parser
my_parser = argparse.ArgumentParser(description="Classify data")

#?- add argument
my_parser.add_argument("path", nargs=1, metavar="path", type=str, help="The path to the csv file data")
my_parser.add_argument("classlabel", nargs=1, metavar="classlabel", type=str, help="Class Label of the file data")
my_parser.add_argument("fillna", nargs=1, metavar="fillna_option", type=str, default='B', help="Fill missing values option: A for depending on general data, B for depending on class label, default: B")
my_parser.add_argument("-normal", nargs=1, metavar="normalization", type=str, default=["False"], help="Normalization: True or False, default: None")
my_parser.add_argument("-discret", nargs=1, metavar="discret_type", type=int, default=[None], help="Discretization type: 0 for equal-width, 1 for equal-frequency, 2 for based entropy, default: None")
my_parser.add_argument("-bins", nargs=1, metavar="discret_bins_nb", type=int, default=[None], help="integer describing which number of bins desired, default: None")
my_parser.add_argument("train", nargs=1, metavar="train_size", type=float, default=0.75, help="Train size: float value in [0, 1] represent the proportion of the dataset to include in the train split, default: 0.75")
my_parser.add_argument("model", nargs=1, metavar="model_type", type=str, help="Model type: string value {'nb': naive bayes model, 'tree': tree decision representation}")
my_parser.add_argument("builtin", nargs=1, metavar="builtin", type=str, default=["False"],help="Builtin version: True for builtin version or False for own version, default: None")
my_parser.add_argument("-pep", nargs=1, metavar="pep", type=str, default=["False"], help="Post Pessimistic Pruning : True for doing a pep or False for not doing , default: None")
my_parser.add_argument("-depth", nargs=1, metavar="max_depth", type=int, default=[None], help="the maximum depth of the tree, integer, default: None (no limit)")
my_parser.add_argument("-fleaf", nargs=1, metavar="min_samples_leaf", type=float, default=[None], help="the minimum samples at leaf for type float, default: None (no limit)")
my_parser.add_argument("-ileaf", nargs=1, metavar="min_samples_leaf", type=int, default=[1], help="the minimum samples at leaf for type integer, default: None (no limit)")


#?-execute parse_args()
args = my_parser.parse_args()
#for boolean value
args.normal[0] = True if args.normal[0] == "True" else False if args.normal[0] == "False" else args.normal[0]
args.builtin[0] = True if args.builtin[0] == "True" else False if args.builtin[0] == "False" else args.builtin[0]
args.pep[0] = True if args.pep[0] == "True" else False if args.pep[0] == "False" else args.pep[0]
print(vars(args))

path_csv = args.path[0]
fillna_option = args.fillna[0]
classlabel = args.classlabel[0]
normalization = args.normal[0]
discret_type = args.discret[0]
discret_bins_nb = args.bins[0]
train_size = args.train[0]
model_type = args.model[0]
builtin = args.builtin[0]
pep = args.pep[0]
max_depth = args.depth[0]
min_samples_leaf = args.fleaf[0] if args.fleaf[0] is not None else args.ileaf[0]

try:
    argChecker(path_csv, fillna_option, classlabel, normalization, discret_type, discret_bins_nb, train_size, model_type, builtin, pep, max_depth, min_samples_leaf)
    execute(path_csv, fillna_option, classlabel, normalization, discret_type, discret_bins_nb, train_size, model_type, builtin, pep, max_depth, min_samples_leaf)
except Exception as e:
    print(e)
    sys.exit()

#CLI use example, run:
#python parsing_perso.py mushrooms.csv class A 0.5 nb True -discret 2 -bins 3 -pep True