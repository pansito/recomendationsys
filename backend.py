from flask import Flask, render_template, jsonify, request
from Model_training import get_recommendations

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/recomendacion', methods=['POST'])
def recomendacion():
    userId = request.json['userId']
    recommendations = get_recommendations(int(userId))
    #print(userId, type(userId))
    return jsonify(recommendations)

if __name__ == "__main__":
    app.run(debug=True)



