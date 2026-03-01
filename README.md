# Student ML Project on CyberSecurity
This project is made by student of the DSTI school.
The objective is to develop a ML model to predict the type of attack.
The project is composed of 3 main parts:
1. Exploratory Data Analysis.
2. Feature Engineering.
3. Model Development.
The repository includes:
1. A main notebook summarising our work.
2. A website to upload data and get prediction using our best model.
3. Scripts we made for each parts

 # Installation
 You may clone the repository and then create a virtual environment and installing the required libraries. <br>
 Steps are: <br>
 1- Create virtual environnement using
 ```bash
 python -m venv ./name_of_venv
 ```
 2- Activate virtual environnement:
 ```bash
 source ./name_of_venv/bin/activate
 ```
2- Install libraries:

```bash
pip install -r requirements.txt
```
If you plan to run a notebook that import the prince library run this command just affter installing the library:
``` python -m scripts/patch_venv.py
```
Prince library is not up to date with pandas 3.X which creates a problem when using it. This quick patch solve the problem temporarily
# Main Notebook

The main notebook (at the root of the repository) is a summary of all the work done by the team through the differents steps.
It runs from top to bottom and features:
<ol>
<li>EDA part
    <ol>
    <li> Presentation of the dataset and the features. <br>
         A quick summary of the data describing data type, presence of missing values etc. </li>
    <li> Analysis of the features <br>
         Short analysis of the features and how they are related to the attack types (distrubtion per attack type)</li>
    </ol>
</li>
</ol>

# Website
We provice a website interface that can be used to upload cyber-attack dataset similar to the one used for the training of our model.
Because the website was developed on another framework. A specific virtual environment must be used. To create it, use the following command:
 ```bash
python -m venv ./name_of_website_venv
source ./name_of_website_venv/bin/activate
pip install -r web_requirements.txt
```

To run the website, use the following command on the terminal
```python
streamlit run Website/ui.py
```
Then open the [following link](http://localhost:8501).

