from flask import Flask, render_template, request
import pickle 
app=Flask(__name__)

tokenizer=pickle.load(open("models/cv.pkl","rb"))
model=pickle.load(open("models/clf.pkl","rb"))

@app.route("/")
def home():
    #text=""
    #if request.method == "POST":
    #   text = request.form.get("email-content")
    return render_template("index.html")#, text=text)


@app.route("/predict", methods=["POST"])
def predict():
    
    email_text=request.form.get("email-content")
    tokenized_email=tokenizer.transform([email_text])
    predictions=model.predict(tokenized_email)
    if predictions ==1:
        predictions=1
    else:
        predictions=-1
    return render_template("index.html", predictions=predictions, email_text=email_text)


if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8080, debug=True)