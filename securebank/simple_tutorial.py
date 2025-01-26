from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def hello_world():
    """
    Index route that returns a welcome message.
    Returns:
    Response with a welcome message.
    """
    return "Hello World!"

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)