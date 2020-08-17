import torch
from torch.utils.data import DataLoader

from config import ENCODING_PATH, IMAGES_PATH, MODEL_PATH
from dataset import ProjectDataset
from models.DAE import Encoder
import json
from pathlib import Path

from utils import get_device

if __name__ == "__main__":
    device = get_device()
    path_to_model = Path(MODEL_PATH)
    path_to_images = Path(IMAGES_PATH)
    graph_dump_path = Path(ENCODING_PATH)

    state_dict = torch.load(path_to_model, map_location=torch.device(device))
    encoder = Encoder()
    encoder.load_state_dict(state_dict['encoder'])
    encoder.to(device)
    dataset = ProjectDataset(path_to_images, device=device)
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