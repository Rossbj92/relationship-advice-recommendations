
from flask import Flask, request, render_template
import recommender as rec

app = Flask(__name__)  # create instance of Flask class

@app.route('/', methods=["POST", "GET"])
def home():
    predictions = rec.recommend(request.form.get('problem'))
    text = request.form.get('problem')
    return render_template('elements.html',
                           text=text,
                           prediction=predictions
                          )

if __name__ == '__main__':
    app.run(debug = False)



