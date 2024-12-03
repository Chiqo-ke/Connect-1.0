# app.py

from flask import Flask, render_template, Response
import sign

app = Flask(__name__)

@app.route('/')
def index():
    # Render the index page where you can start sign recognition
    return render_template('index.html')

@app.route('/run-sign-recognition')
def run_sign_recognition():
    # Start the sign recognition function from sign.py
    sign.run_sign_recognition()
    return "Sign recognition process ended."

if __name__ == '__main__':
    app.run(debug=True)
