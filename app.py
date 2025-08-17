from flask import Flask,render_template,request
import joblib
import pandas as pd
app=Flask(__name__)

@app.route("/",methods=["GET"])
def home():
    return render_template("home.html")

@app.route("/form",methods=["GET"])
def form():
    return render_template("form.html")
model=joblib.load("modelpredict.pkl")
@app.route("/predict",methods=["POST"])
def predict():
    timespentalone=float(request.form["Time_spent_Alone"])
    stagefear=request.form["Stage_fear"]
    socialevent=float(request.form["Social_event_attendance"])
    goingoutside=float(request.form["Going_outside"])
    drained=request.form["Drained_after_socializing"]
    friends=float(request.form["Friends_circle_size"])
    posts=float(request.form["Post_frequency"])
    data = {
    "Time_spent_Alone": [timespentalone],
    "Stage_fear": [stagefear],
    "Social_event_attendance": [socialevent],
    "Going_outside": [goingoutside],
    "Drained_after_socializing": [drained],
    "Friends_circle_size": [friends],
    "Post_frequency": [posts]
}
    features=pd.DataFrame(data)
    prediction=model.predict(features)
    result=""
    if(prediction==0):
        return render_template("form.html",result="Extrovert")
    else:
        return render_template("form.html",result="Introvert")

    
# Time_spent_Alone	Stage_fear	Social_event_attendance	Going_outside	Drained_after_socializing	Friends_circle_size	Post_frequency








if(__name__=="__main__"):
    app.run(debug=True)