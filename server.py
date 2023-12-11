from flask import Flask, request, jsonify
from dockerInference import LLM
import argparse

app = Flask(__name__)

parser = argparse.ArgumentParser(
                    prog='Yum',
                    description='This is a server for the Yum chatbot LLM',
                    epilog='plz help')

parser.add_argument('--early_stopping', type=bool, default=True)
parser.add_argument('--bos_token_id', type=int, default=None)
parser.add_argument('--eos_token_id', type=int, default=None)
parser.add_argument('--pad_token_id', type=int, default=None)
parser.add_argument('--do_sample', type=bool, default=True)
parser.add_argument('--top_k', type=int, default=30)
parser.add_argument('--top_p', type=float, default=0.95)
parser.add_argument('--temperature', type=float, default=0.5)
parser.add_argument('--max_new_tokens', type=int, default=1152)
parser.add_argument('--repetition_penalty', type=float, default=1.2)
args = parser.parse_args()

classifier = LLM(
    args.early_stopping,
    args.bos_token_id,
    args.eos_token_id,
    args.pad_token_id,
    args.do_sample,
    args.top_k,
    args.top_p,
    args.temperature,
    args.max_new_tokens,
    args.repetition_penalty
)

@app.route('/predict', methods=['POST'])
def predict_intent():
    instruction = request.json['instruction']
    systemMessage = request.json['systemMessage']

    response,input_text = classifier.predict(instruction, systemMessage)
    print("INPUT_TEXT to llm: \n",input_text, end='\n==================\n')

    return jsonify({"data":response})

if __name__ == '__main__':
    app.run(host= '0.0.0.0', debug=True)