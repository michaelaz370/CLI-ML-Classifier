import pandas as pd
from sklearn.model_selection import train_test_split
import os
from datetime import datetime
import joblib

from preprocessing import Preprocessing
from naivebayes import NaiveBayesClassifier
from treedecision import TreeDecisionClassifier
from evaluation import Evaluation



def splitData(df, train_size):
  """
  :param df: data frame
  :param train_size: float value in [0, 1] represent the proportion of the dataset to include in the train split
  :returns feature train set, feature test set, target train set, target test set
  """
  train, test = train_test_split(df, random_state=0, train_size=train_size)
  return train, test


def execute(path_csv, fillna_option, classlabel, normalization, discret_type, discret_bins_nb, train_size, model_type,
            builtin, pep=True, max_depth=None, min_samples_leaf=1):
  """
  :param path_csv; string path to the data file
  :param fillna_option: option to fill missing values: A for general, B for class label
  :param classlabel: class label
  :param normalization: boolean
  :param discret_type: integer  describing which discretization type desired : 0 for equal-width, 1 for equal-frequency, 2 for based entropy, default: None
  :param discret_bins_nb: integer  describing which number of bins desired, default: None
  :param train_size: float value in [0, 1] represent the proportion of the dataset to include in the train split
  :param model_type: string value {nb: naive bayes model, tree: tree decision representation}
  :param pep: boolean, default: None
  :param max_depth: the maximum depth of the tree, integer, default: None (no limit)
  :param min_samples_leaf: the minimum sample at leaf ,or integer or float number in [0,1], default: 1
  """
  #Create backup directory
  try:
    state = joblib.load("state.obj")
  except Exception as e:
    state = {"id": 0}

  state['id'] += 1

  now = datetime.now().strftime("-%Y%m%d-%Hh%Mmin%Ssec")
  path = "backup" + now + "-" + str(state['id'])
  if not os.path.exists(path):
    os.makedirs(path)
  path += '/'

  joblib.dump(state, "state.obj")



  df = pd.read_csv(path_csv, na_values=["?"])
  filename = path_csv.split('/')[-1].split('.')[0]

  #Preprocessing
  prepro = Preprocessing(df, fillna_option, classlabel, filename, normalization, discret_type, discret_bins_nb, path)

  #Split data # A CHECKER: IL FAUDRAIT PAS PLUTOT METTRE prepro.df plutot que df ??????
  train, test = splitData(df, train_size)

  #Build model and save model
  if model_type == "nb":
    classifier = NaiveBayesClassifier(train, classlabel, prepro.enc_dec_dict, builtin, path)
  else:
    classifier = TreeDecisionClassifier(train, classlabel, prepro.enc_dec_dict, pep, max_depth, min_samples_leaf, builtin, path)

  classifier.saveModel()

  #Evaluation
  eval = Evaluation("class", train, test, classifier, path)
  eval.script()
