# from utils.log import Log
# from models.ToyModel import ToyModel
# from models.ModelRunner import ModelRunner

if __name__ == '__main__':
    # logger = Log('test')
    # for i in range(10):
    #     logger.write(str(i))
    #     print(i)
    #     exit()
    # logger.f.close()

    # runner = ModelRunner('test', model=ToyModel())
    # runner.set_optimizer()
    # runner.save_model(0, 0)
    #
    # runner2 = ModelRunner('test', model=ToyModel())
    # runner2.set_optimizer()
    # runner2.load_model(0)
    # print(runner2.model)
    # print(runner2.optimizer)
    import torch
    from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

    # OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
    # import logging
    #
    # logging.basicConfig(level=logging.INFO)

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenized input
    text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
    tokenized_text = tokenizer.tokenize(text)

    # Mask a token that we will try to predict back with `BertForMaskedLM`
    masked_index = 8
    tokenized_text[masked_index] = '[MASK]'
    assert tokenized_text == ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', '[MASK]', 'was', 'a',
                              'puppet', '##eer', '[SEP]']

    # Convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # indexed_tokens = [indexed_tokens, indexed_tokens]
    # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
    segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    # segments_ids = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()

    # If you have a GPU, put everything on cuda
    tokens_tensor = tokens_tensor.to('cuda')
    segments_tensors = segments_tensors.to('cuda')
    model.to('cuda')

    # Predict hidden states features for each layer
    with torch.no_grad():
        encoded_layers, hid = model(tokens_tensor, segments_tensors)
        print(tokens_tensor.shape)
        print(segments_tensors.shape)
        print(encoded_layers[-1].shape)
        print(hid.shape)
    # We have a hidden states for each of the 12 layers in model bert-base-uncased
    assert len(encoded_layers) == 12
    print(model.device)