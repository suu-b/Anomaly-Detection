### Setup for the app
Run the following command in your python virtual environment:

```
    pip install -r requirements.txt
    python model.py
    streamlit run app.py
```
The first command will install all the required modules. The next command will train a LOF model using the data provided in the repo at location: `/data/train1.csv` and freeze a the model as a artifact using `joblib` library.
Lastly, we will run a streamlit app which will open an interface for the user to enter some data and check if it is an anomaly or not. The interface is as such:

![UI](../README_files/streamlit_app_1.png)

***

Note: The app is merely part of this whole exploratory repo of me at ML. It may serve value for some and not for the others.