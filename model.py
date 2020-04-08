import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        # saving the parameter for further use
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        #creating embedding layer
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        
        #creating lstm layer
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        #creating linear layer to map output lstm to all the words of vocabulary
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    

    
    
    
    
    def forward(self, features, captions):
        #removing the last word and creating embedded word vector
        captions = self.word_embeddings(captions[:,:-1])
        
        
        embed = torch.cat((features.unsqueeze(1), captions), dim=1)
        
        
        self.hidden = (torch.zeros(1, embed.size(0),  self.hidden_size, device="cuda"), 
                  torch.zeros(1, embed.size(0),  self.hidden_size, device="cuda"))
        
        
        
        output, self.hidden = self.lstm(embed, self.hidden)
        
        outputs = self.linear(output)
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        """
        Accepts pre-processed image tensor (inputs) and returns predicted sentence 
        (list of tensor ids of length max_len) 
        """
        res = []
        self.hidden = (torch.zeros(1, 1,  self.hidden_size, device="cuda"), 
                  torch.zeros(1, 1,  self.hidden_size, device="cuda"))
        for i in range(max_len):
            output, self.hidden = self.lstm(inputs, self.hidden)         # hiddens: (1, 1, hidden_size)
            
            output = self.linear(output.squeeze(1))       # outputs: (1, vocab_size)
            
            _,predicted = output.max(dim=1)                    # predicted: (1, 1)
            
            
            if(predicted.item() == 1):
                res.append(predicted.item())
                break                # end tokken is come , no need to proceed
            res.append(predicted.item())
            
            inputs = self.word_embeddings(predicted)             # inputs: (1, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (1, 1, embed_size)
        return res
