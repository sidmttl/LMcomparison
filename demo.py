import argparse
import pickle
import torch

from attention import *
from mamba import *
from xlstm import *



def main():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Script to Run Large Language Models')

    # Model argument
    parser.add_argument(
        '-m', '--model',
        choices=['attention', 'mamba', 'xlstm'],
        required=True,
        help='Choose an AI model: attention, mamba, or xlstm'
    )

    # Context argument
    parser.add_argument(
        '-c', '--context',
        type=str,
        required=False,
        help='Context to be used for the Language Model'
    )

    # language argument
    parser.add_argument(
        '-l', '--lang',
        choices=['eng', 'hi'],
        required=False,
        help='Set the Language of the model. Defaults to eng',
    )

    # Parse the arguments
    args = parser.parse_args()

    with open('stoi_shake.pkl', 'rb') as f:
        stoi = pickle.load(f)

    with open('itos_shake.pkl', 'rb') as f:
        itos = pickle.load(f)

    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    # Use the arguments
    print(f"Selected Model: {args.model}")

    torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Add context
    if args.context:
        context = torch.tensor(encode("the night class is"), dtype=torch.long)
        context = torch.unsqueeze(context, 0)
    else:
        context = torch.zeros((1, 1), dtype=torch.long)


    # Run respective model
    if args.model == 'attention':
        gpt_model = torch.load('./models/attention_best_model.pt')
        out = decode(gpt_model.generate(context, max_new_tokens=500)[0].to('cpu').tolist())
        print(out)


    elif args.model == 'mamba':
        mamba_model = torch.load('./models/mamba_best_model.pt')
        out = decode(mamba_model.generate(context, max_new_tokens=500)[0].to('cpu').tolist())
        print(out)


    elif args.model == 'xlstm':
        xlstm_model = torch.load('./models/xlstm_best_model.pt')
        print(context, end="")
        for _ in range(500):
            context, out = xlstm_model.generate(context, max_new_tokens=1)
            print(decode(out[0].to('cpu').tolist()), end="")

if __name__ == '__main__':
    main()
