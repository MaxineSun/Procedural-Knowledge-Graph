import openai

openai.api_key = "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"

try:
    models = openai.Model.list()
    print("Connection with OpenAI established successfully!")
except Exception as e:
    print("Error connecting to OpenAI: {}".format(e))
