
MAX_LENGTH = 15

def train(train_iter, val_iter, model, criterion, optimizer, num_epochs):

    for epoch in range(num_epochs):
        model.train()
        for batch in train_iter:
            print(batch.size())
            source = batch.src
            target = batch.trg

            model.zero_grad()

            scores = model(source, target)

            loss = 

