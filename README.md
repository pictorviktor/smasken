# smasken

To get the libraries to work on python, follow the below steps:

1. Create a new folder to place your libraries in such as "Numpy_projects"

2. Navigate to your new folder in the terminal

3. Create the file you want to import your libraries in.

4. Create a virtual environment(venv) using your terminal with the following command:

$ python<your version of python> -m venv <Your name of choice for your venv>

5. Click yes in the right corner when asked to.

6. You now see your venv within your folder to the left. You need to activate your venv using the following command:

$ source <the name of your venv>/bin/activate


7. You can see that your venv is activated in the terminal by the parentheses in the beginning of the line stating your venv.

8. You are now ready to install the packages to your folder within your venv. An example could be:

$ pip3 install numpy or pip3 install pandas

9. All you need to do now is to import the library into your file. It looks different for different libraries but one example is:
"import numpy as np"

10. You are done!
