import torch
import torch.nn as nn
from torch.autograd import Variable
import dataset_process
from network import Net
import captcha_generate

# Hyper Parameters
num_epochs = 10
batch_size = captcha_generate.BATCH_SIZE
learning_rate = 0.001


def main():
    avgLoss = 0.0
    loss1, loss2, loss3, loss4 = 0.0, 0.0, 0.0, 0.0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net().to(device=device)
    # net.to(device)
    net.train()
    print('init net')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    train_dataloader = dataset_process.get_train_data_loader()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_dataloader):
            images = Variable(images).to(device=device)
            labels = Variable(labels.long()).to(device=device)
            labels1, labels2, labels3, labels4 = torch.max(labels[:, 0], dim=1)[1], torch.max(labels[:, 1], dim=1)[1], \
                                                 torch.max(labels[:, 2], dim=1)[1], torch.max(labels[:, 3], dim=1)[1]
            predict_labels1, predict_labels2, predict_labels3, predict_labels4 = net(images)
            loss1 = criterion(predict_labels1, labels1)
            loss2 = criterion(predict_labels2, labels2)
            loss3 = criterion(predict_labels3, labels3)
            loss4 = criterion(predict_labels4, labels4)
            loss = loss1 + loss2 + loss3 + loss4
            avgLoss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 10 == 0:
                print("epoch:", epoch, "step:", i, "loss:", avgLoss / 10)
                avgLoss = 0
            if (i + 1) % 100 == 0:
                torch.save(net.state_dict(), "./model.pkl")  # current is model.pkl
                print("save model")
        print("epoch:", epoch, "step:", i, "loss:", avgLoss / 10)
    torch.save(net.state_dict(), "./model.pkl")  # current is model.pkl
    print("save last model")


if __name__ == "__main__":
    main()
