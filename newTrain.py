import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from changeModel import RohitConv
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)
from diceLoss import DiceLoss

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 256  # 768 originally
IMAGE_WIDTH = 256  # 768 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "Done_Train/"
TRAIN_MASK_DIR = "Please_Val/"
VAL_IMG_DIR = "Done_Val/"
VAL_MASK_DIR = "Try_Final_Im_Done/"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        # with torch.cuda.amp.autocast():
        # data = data.squeeze()
        # print(max(data))
        # with open('test.txt', 'w') as f:
        #     f.write(str(data))
        #     f.write('\n')
        # close('test.txt')
        predictions = model(data)
        loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=1.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    preModel = RohitConv().to(DEVICE)
    # model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = DiceLoss()
    loss_fn_pre = nn.BCEWithLogitsLoss()
    optimizer_pre = optim.Adam(preModel.parameters(), lr = LEARNING_RATE)
    optimizer = optim.Adam(preModel.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpointNew.pth.tar"), preModel)

    # print(train_loader.shape)
    # check_accuracy(val_loader, preModel, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):

        train_fn(train_loader, preModel, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": preModel.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, preModel, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, preModel, folder="saved_new_images/", device=DEVICE
        )


if __name__ == "__main__":
    main()