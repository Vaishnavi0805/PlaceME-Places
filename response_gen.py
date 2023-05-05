import execution
import re
import random

def gen_response(userText):
    # userText = "I'm feeling anxious, so I want to find a spot to be more sad"
    location_list, emotion = execution.location_recommendation(userText)
    # Define patterns and corresponding responses
    patterns = {
        r"Excited": ["It's great to hear that you're feeling excited!", "That's fantastic news!", "I'm so happy to hear that!"],
        r"Joy": ["That's great to hear!", "Wonderful news! You're really making my day.", "Congratulations! It's always great to hear about people feeling joyful.", "I'm thrilled for you!"],
        r"Neutral": ["I see.", "Alright.", "I understand."],
        r"Sad": ["I'm sorry to hear that you're feeling down.", "I'm here for you if you need someone to talk to.", "It's okay to feel sad sometimes."],
        r"Disappointed": ["I'm sorry to hear that you're feeling disappointed.", "That must be tough. ", "I'm here to listen if you need to vent."]
    }

    # Define a function to generate responses
    def respond(message):
        for pattern, responses in patterns.items():
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                return random.choice(responses)
        return "May be I can help you with that"

    First_line=respond(emotion)

    response_str = ""
    for i, response in enumerate(location_list, 1):
        response_str += str(i) + ".\n"
        for item in response:
            response_str += item + "\n"
        response_str += "\n"
    # Convert list of lists to string

    # print(response_str)
    result = First_line + "\n\n" + response_str
    # result = First_line + response_str

    # print(result)
    return result