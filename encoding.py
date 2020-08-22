import torch
from torch.utils.data import DataLoader

from config import MODEL_PATH
from dataset import ProjectDataset
from models.DAE import Encoder
import json
from pathlib import Path

from utils import get_device

def encode(model_path):
    device = get_device()
    path_to_model = Path(model_path) / Path('model.pth')
    graph_dump_path = Path(model_path)/ Path('encoding.json')
    config_path = Path(model_path) / Path('config.json')
    with config_path.open() as f:
        config = json.load(f)
    images_path = config['IMAGES_PATH']
    out_channels = config['out_channels']
    dims = config['dims']

    state_dict = torch.load(path_to_model, map_location=torch.device(device))
    encoder = Encoder(dims, out_channels)
    encoder.load_state_dict(state_dict['encoder'])
    encoder.to(device)
    dataset = ProjectDataset(images_path, device=device)
    data_loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)

    points = []
    for i, datum in enumerate(data_loader):
        image_path = datum['path'][0]
        image = datum['img']
        vector = encoder(image)
        z = vector.flatten()
        points.append({'z': z.tolist(), 'path': image_path})
    with graph_dump_path.open('w') as f:
        json.dump(points, f)

if __name__ == "__main__":
    encode(MODEL_PATH)