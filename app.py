from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import zipfile


app = Flask(__name__)
scaler = pickle.load(open('scaler_model.pkl', 'rb'))

zip_filename = "model.zip"

# Extract the model file from the ZIP archive
with zipfile.ZipFile(zip_filename, 'r') as archive:
    file_list = archive.namelist()
    if len(file_list) == 1:
        model_filename = file_list[0]
        with archive.open(model_filename) as model_file:
            model = pickle.load(model_file)
            print("Model loaded successfully.")
    else:
        print("Expected only one file in the archive, but found: %s" % file_list)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # get the data from the json request

    user_input = {}
    user_input["value_eur"] = request.get_json()['value_eur']
    user_input["release_clause_eur"] = request.get_json()['release_eur']
    user_input["cat_age"] = request.get_json()['cat_age']
    user_input["potential"] = request.get_json()['potential']
    user_input["movement_reactions"] = request.get_json()['movement_reactions']

    # make prediction

    user_input_dataframe = pd.DataFrame([user_input])
    user_input_dataframe = scaler.transform(user_input_dataframe)


    prediction = model.predict(user_input_dataframe)
    int_prediction = int(round(prediction[0], 0))


    predictions_by_estimators = [int(tree.predict(user_input_dataframe)[0]) for tree in model.estimators_]
    estimates_dictionary = {}

    for estimate in predictions_by_estimators:
        if estimate in estimates_dictionary:
            estimates_dictionary[estimate] += 1
        else:
            estimates_dictionary[estimate] = 1
    
    frequency = estimates_dictionary[int_prediction]
    total = len(predictions_by_estimators)

    # calculating confidence as the number of estimator that voted for the in prediction

    confidence = round((frequency/total) * 100, 2)

    return jsonify({'prediction': int_prediction, 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True)