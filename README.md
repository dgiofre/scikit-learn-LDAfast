# scikit-learn-LDAfast
An optimized cython version of LDA model implemented in scikit-learn

## Installation
I just show the procedure for a Mac using pyenv (feel free to use your python library and environment manager):

```sh
brew install readline xz
brew install pyenv pyenv-virtualenv

pyenv install 3.7.9
pyenv virtualenv 3.7.9 fastLDA_3.7

cd {YOUR_DIR_PATH}/scikit-learn-LDAfast
pyenv local fastLDA_3.7
```

Now open a new terminal and navigate in the code directory.
You should have "fastLDA" environment active. 
If so please launch the following commands

```sh
pip install -r requirements.txt
python -m ipykernel install --user --name fastLDA_3.7 --display-name "fast LDA  3.7"
```

Now it is time to compile your Cythonic code:
```sh
cd LDAfast/clib
python setup.py build_ext --inplace
```

Well done, you are ready to use the fastLDA method!

## API

See the example notebook on how you can call the fastLDA method and the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html) for the use, args and the parameters definition.

Good decomposition!


