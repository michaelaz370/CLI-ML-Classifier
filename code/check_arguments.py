import os
import pandas as pd

def argChecker(path, fillna, classlabel, normal, discret, bins, train, model, builtin, pep, depth, leaf):
    check_path_csv(path)
    checkClassLabel(path, classlabel)

    option_dict = {"fillna": ['A', 'B'], "discret": [0, 1, 2], "model": ["nb", "tree"]}

    checkDiscretization(option_dict["discret"], discret, bins)

    checkFillna(option_dict["fillna"], fillna)

    checkBoolean(normal, builtin)

    checkModel(option_dict["model"], model, depth, leaf, pep, builtin)

    checkTrainSize(train)



def check_path_csv(path):
    if not os.path.exists(path):
        raise Exception("The path specified in path argument doesn't exist")

    elif not path.endswith(".csv"):
        raise Exception("The type file in path argument is not csv")

def checkClassLabel(path ,classlabel):
    df = pd.read_csv(path, na_values=['?'])
    if classlabel not in df.columns:
        raise Exception("The class label provided doesn't exist in the data frame")

def checkDiscretization(types, discret, bins):
    if discret != None:
        if discret not in types:
            raise Exception("'{0}' is not a discret type option, allowed options for discret argument are : {1}".format(discret, types))

    if type(bins) is not int:
        raise Exception("'{0}' is not a integer value, type an integer value for bins argument".format(bins))


def checkFillna(options, fillna):
    if fillna not in options:
        raise Exception("'{0}' is not a fillna option, allowed options for fillna argument are : {1}".format(fillna, options))


def checkModel(models, model, depth, leaf, pep, builtin):
    if model not in models:
                raise Exception("'{0}' is not a model type option, allowed options for model argument are : {1}".format(model, models))
    elif model == "tree" and not builtin:
        if type(pep) is not bool:
            raise Exception("You choosed a non builtin tree decision, '{0}' is not a boolean value, type or True or False for pep argument".format(pep))
        if depth != None or leaf != None:
            if depth != None and type(depth) is not int:
                raise Exception("'{0}' is not a integer value, type an integer value for depth argument".format(depth))
            if leaf != None:
                if type(leaf) is int:
                    if leaf < 1:
                        raise Exception("leaf argument can not be an integer less than 1")

                elif type(leaf) is float:
                    if not (leaf >= 0 and leaf <= 1):
                        raise Exception("leaf argument can not be a float less than 0 or more than 1".format(leaf))

                else:
                    raise Exception("leaf argument has to be or an integer >= 1 or a float in [0, 1]")
        else:
            raise Exception("You choosed a non builtin tree decision, so you have to specify a leaf or a depth argument")

def checkBoolean(normal, builtin):
    if normal != None and type(normal) is not bool:
        raise Exception("'{0}' is not a boolean value, type or True or False for normal argument".format(normal))

    if type(builtin) is not bool:
        raise Exception("'{0}' is not a boolean value, type or True or False for builtin argument".format(builtin))


def checkTrainSize(train):
    if type(train) is not float or not (train >= 0 and train <= 1):
        raise Exception("'{0}' is not a correct value, type a float value between 0 and 1 for train argument".format(train))

