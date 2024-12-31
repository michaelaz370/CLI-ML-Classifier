## Author:
	Michael Azeroual 342532421

## About this project:
In this project I built a classification and clustering CLI system working on any dataset using differents Machine Learning
models built from scratch : decision tree, naive-bayes.
The data file can contain both continuous and categorical features. In addition, there may be records with missing values,
which is addressed as part of the data cleansing process.
dataset example:
    class label: p = poisonous and e = edible.
    for more informations about the data: https://www.kaggle.com/datasets/uciml/mushroom-classification


## Installation
    	You will need an IDE for phyton. (Pycharm3.9 is required)
    	all the pip installations can be acquired  from the requirements.txt file.

    	Python 3.6.9 was used to create the project


## pip freeze:
	certifi==2021.5.30
	cycler @ file:///home/conda/feedstock_root/build_artifacts/cycler_1635519461629/work
	joblib @ file:///tmp/build/80754af9/joblib_1613502643832/work
	kiwisolver @ file:///D:/bld/kiwisolver_1610099971815/work
	matplotlib @ file:///D:/bld/matplotlib-suite_1611858808344/work
	mkl-fft==1.3.0
	mkl-random==1.1.0
	mkl-service==2.3.0
	numpy @ file:///C:/ci/numpy_and_numpy_base_1603480701039/work
	olefile @ file:///home/conda/feedstock_root/build_artifacts/olefile_1602866521163/work
	pandas @ file:///C:/ci/pandas_1608056614942/work
	Pillow @ file:///D:/bld/pillow_1630696770108/work
	pyparsing @ file:///home/conda/feedstock_root/build_artifacts/pyparsing_1652235407899/work
	PyQt5==5.12.3
	PyQt5_sip==4.19.18
	PyQtChart==5.12
	PyQtWebEngine==5.12.1
	python-dateutil @ file:///tmp/build/80754af9/python-dateutil_1626374649649/work
	pytz==2021.3
	scikit-learn @ file:///C:/ci/scikit-learn_1622739439730/work
	scipy @ file:///C:/ci/scipy_1597675683670/work
	six @ file:///tmp/build/80754af9/six_1644875935023/work
	threadpoolctl @ file:///Users/ktietz/demo/mc3/conda-bld/threadpoolctl_1629802263681/work
	tornado @ file:///D:/bld/tornado_1610094881553/work
	wincertstore==0.2


## Project file structure:
	In my Project there are 7 files:
	* preprocessing.py which contains the Preprocessing class in order to execute all preprocessing tasks.
	* naivebayes.py which contains the NaiveBayesClassifier class in order to build Naïve Bayes Classifier
  	builtin and own implemented models.
	* treedecision.py which contains the TreeDecisionClassifier and Node classes in order to build Tree Decision
	Classifier builtin and own implemented models.
	* evaluation.py which contains the Evaluation class for evaluation tasks (confusion matrix, classification report,
	roc curve etc...)
	* proc.py contains a splitData function for splitting in train and test set and a execute function 
	to execute the software
	* parsing_perso.py for cli interface
	* check_arguments.py for checking arguments and raising errors


## use:
	* To run all experiments(script), use: 
		- linux: bash loop.sh
		- windows: powershell loop.sh
	* To run individual experiment use: python parsing_perso.py <arguments>
	example:python parsing_perso.py mushrooms.csv class A 0.5 nb True -discret 2 -bins 3 -pep True

	For more information on arguments use : python parsing_perso.py -h