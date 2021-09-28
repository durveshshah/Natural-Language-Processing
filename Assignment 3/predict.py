"""
Created on: 2021-02-06
Author: duytinvo
"""
import torch
from utils.core_nns import RNNModel
from utils.data_utils import Txtfile, Data2tensor
from utils.data_utils import SOS, EOS, UNK
from utils.data_utils import SaveloadHP


class LMInference:
    def __init__(self, arg_file="./results/lm.args", model_file="./results/lm.m"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args, self.model = self.load_model(arg_file, model_file)

    def load_model(self, arg_file="./results/lm.args", model_file="./results/lm.m"):
        """
        Inputs:
            arg_file: the argument file (*.args)
            model_file: the pretrained model file
        Outputs:
            args: argument dict
            model: a pytorch model instance
        """
        args, model = None, None
        #######################
        # YOUR CODE STARTS HERE
        args = SaveloadHP.load(arg_file)
        model = RNNModel(args.model, len(args.vocab.w2i), args.emsize, args.nhid, args.nlayers, args.dropout, args.tied,args.bidirect)
        model_load_state = model.load_state_dict
        model_load_state(torch.load(model_file))
        # YOUR CODE ENDS HERE
        #######################
        return args, model

    def generate(self, max_len=1000):
        """
        Inputs:
            max_len: max length of a generated document
        Outputs:
             the text form of a generated document
        """
        doc = [SOS]
        #######################
        # YOUR CODE STARTS HERE
        i = 0
        newlist = list()
        i2w = self.args.vocab.i2w
        while i < max_len:

            if i == 0:
                hid = self.model.init_hidden(1)
                input = torch.randint(0, len(i2w), (1, 3))
                out, hid = self.model.forward(input, hid)
                prob, label = self.model.inference(out)
                word_string = i2w[label[0][0][0].item()]
                if word_string == '</s>':
                    break
                newlist.append(word_string)


            elif i > 0:
                label_to_rehape = torch.reshape(label, (1, -1))
                out, hid = self.model.forward(label_to_rehape, hid)
                prob, label = self.model.inference(out)
                word_string = i2w[label[0][0][0].item()]
                if word_string == '</s>':
                    break
                newlist.append(word_string)
            i += 1

        doc += newlist
        # YOUR CODE ENDS HERE
        #######################
        doc += [EOS]
        return " ".join(doc)

    def recommend(self, context="", topk=5):
        """
        Inputs:
            context: the text form of given context
            topk: number of recommended tokens
        Outputs:
            A list form of recommended words and their probabilities
                e,g, [('i', 0.044447630643844604),
                     ('it', 0.027285737916827202),
                     ("don't", 0.026111900806427002),
                     ('will', 0.023868300020694733),
                     ('had', 0.02248169668018818)]
        """
        rec_wds, rec_probs = [], []
        #######################
        # YOUR CODE STARTS HERE
        newlist = list()
        string_words = context.split()
        w2i = self.args.vocab.w2i
        i = 0
        while i < len(string_words):
            newlist.append(w2i[string_words[i]])
            convert_tensor = torch.tensor(newlist)
            i += 1

        hidden = self.model.init_hidden(1)
        reshape_tensor = torch.reshape(convert_tensor, (1, -1))
        output, hidden = self.model.forward(reshape_tensor, hidden)
        prob, label = self.model.inference(output, topk)
        for j in range(topk):
            for k in label[0, -1].tolist():
                rec_wds.append(self.args.vocab.i2w[k])
            rec_probs.append(prob[0][-1][j].item())
        # YOUR CODE ENDS HERE
        #######################
        return list(zip(rec_wds, rec_probs))


if __name__ == '__main__':
    arg_file = "./results/lm.args"
    model_file = "./results/lm.m"
    lm_inference = LMInference(arg_file, model_file)

    max_len = 20
    doc = lm_inference.generate(max_len=max_len)
    print("Random doc: {}".format(doc))
    context = "i went to school"
    topk = 5
    rec_toks = lm_inference.recommend(context=context, topk=topk)
    print("Recommended words of {} is:".format(context))
    for wd, prob in rec_toks:
        print("\t- {} (p={})".format(wd, prob))
    pass
