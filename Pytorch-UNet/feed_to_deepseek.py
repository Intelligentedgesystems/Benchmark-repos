from openai import OpenAI

client = OpenAI(api_key="sk-62dcf8b47a7d498ba891877426840f70", base_url="https://api.deepseek.com")

def read_messages_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        messages = []
        messages.append({"role": "system", "content": "You are a helpful assistant."})
        messages.append({"role": "user", "content": "I am going to provide to you the python project code in heirarchial manner, like first the entry point of code till any function is called and after that the function code and after returning back to main module, hence it forms a heirarchial. Each call would be somewhat like {function_name: , filename: location of file, lineno: line which is executed, args: [arg_name: {type: , shape: }] , lines: code which are executed one by one, executed_lines: line numbers which are executed , executed_function_lines: line number of functions that are executed till now, extra_calls: , return_value: {type: shape: }} Using these inputs finally create a python pipeline which take real time input from gstreamer and output the data from model."})
        for i, line in enumerate(lines):
            if line.strip():
                messages.append({"role": "user", "content": line.strip()})
    return messages

# Path to the .txt file
file_path = 'trace_calls1.txt'

# Read messages from file
messages = read_messages_from_file(file_path)
# print(messages)

# Create the chat completion request
response = client.chat.completions.create(
    model="deepseek-coder",
    messages=messages,
    stream=False
)

# Print the response
print(response.choices[0].message)

