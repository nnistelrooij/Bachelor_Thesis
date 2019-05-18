# Bachelor Thesis Code 2019 Niels van Nistelrooij

This is the repository of the Bachelor Thesis of Niels van Nistelrooij in 2019.

The code is written in such a way, that you only have to change the code in `test.py` to change the model, number of free parameters, which plots are made, etc. The plots show the data of a single experiment, whereas the data that is printed to the terminal is summary data across experiments. The code can for example easily be expanded with a third stimulus selection technique or by implementing GPU integration with NumPy.

Even though quit a lot of effort has been put in the efficiency of the code, running the model with all free parameters 20 times to recreate the figures in the thesis still takes 3 hours on my laptop. So do be aware that it might take a very long time before the program finishes. It must be noted though, that adding plots for every experiment greatly increases the processing time, so it is advised to only print the data to the terminal if you want to replicate the figures.

The code is based on Python 2.7.15, so a good way to understand the code is to port it to Python 3 or another language. It probably works faster and better on these alternative languages.
