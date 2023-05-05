from flask import Flask, render_template, flash, redirect, url_for, request, session, jsonify
import execution, response_gen

# input_text = "I'm feeling anxious, so I want to find a spot to be more sad"
# location_list = execution.location_recommendation(input_text)
# print(location_list)

app = Flask(__name__, static_url_path='/static')


@app.route("/",methods=['GET','POST'])
def home():
    return render_template('home.html')

@app.route("/chatbot",methods=['GET','POST'])
def chatbot():
    return render_template('index.html')

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    # print(userText)
    response = response_gen.gen_response(userText)
    # print(location_list) 
    return response

# def save_response(userText):
#     global userTextResp =userText
#     return redirect(url_for('profile_admin'))


if __name__ == "__main__":
    app.run(debug=True)