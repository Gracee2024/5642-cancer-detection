import warnings
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import sys
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F

try:
    sys.path.insert(0, '.')
    from model.base import Model
    from utils import parse, get_logger, print_dct
finally:
    pass

warnings.filterwarnings('ignore')


class TestDataset(Dataset):
    def __init__(self, data_dir, transform, data_type="test"):
        path2data = os.path.join(data_dir, data_type)
        # labels are in a csv file named train_labels.csv
        csv_filename = "sample_submission.csv"
        path2csv = os.path.join(data_dir, csv_filename)
        self.test_df = pd.read_csv(path2csv)
        # set data frame index to id
        self.test_df.set_index("id", inplace=True)
        # obtain labels from data frame
        self.full_filenames = [os.path.join(path2data, id)+'.tif' for id in list(self.test_df.index)]
        self.transform = transform

    def __len__(self):
        # return size of dataset
        return len(self.full_filenames)

    def __getitem__(self, idx):
        # open image, apply transforms and return with label
        image = Image.open(self.full_filenames[idx])  # PIL image
        image = self.transform(image)
        return image


def test():
    args = parse()
    device = torch.device(args['device'])
    model = Model(args['model'])
    # 加载模型
    checkpoint = f"{args['save_path']}{args['model']['name'].lower()}/" + f'{args["model"]["name"]}.pth'
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.70244707, 0.54624322, 0.69645334), (0.23889325, 0.28209431, 0.21625058))
    ])
    test_data = TestDataset(args['data_dir'], transform)
    test_dataloader = DataLoader(test_data, batch_size=256, shuffle=False)

    model.eval()
    y_preds = []
    with torch.no_grad():
        for imgs in tqdm(test_dataloader):
            imgs = imgs.to(device)
            preds = F.softmax(model(imgs))
            y_pred = preds[:, 1].detach().cpu().numpy()
            y_preds.extend(y_pred.tolist())

    test_data.test_df['label'] = y_preds
    test_data.test_df.to_csv(f"submission_{args['model']['name']}.csv")
    print('done')


if __name__ == '__main__':
    test()
