import torch 
from ggpt.dataloader.base_dataset import BaseDataset

class DemoDataset(BaseDataset):
    def __init__(self, name, ff_path, geo_path, **kwargs):
        """
        A Demo Dataset for demo purpose.
        It contains a single scene.
        It does not need Ground Truth for evaluation.
        """
        super().__init__(**kwargs)
        self.name = name
        self.ff_path = ff_path
        self.geo_path = geo_path
        self.mode = 'demo' 
    
    def load_scene(self, idx):
        assert idx == 0
        ff_data = torch.load(self.ff_path, map_location='cpu')
        geo_data = torch.load(self.geo_path, map_location='cpu')
        scene = {
            'dataset_name': 'demo',
            'scene_name': self.name,
            'ff_pts': ff_data['points'], # (N,H,W,3)
            'ff_conf': ff_data['points_conf'], # (N,H,W)
            'geo_pts': geo_data['points'], # (N,H,W,3)
            'geo_msks': geo_data['point_masks'], # (N,H,W)
            'images': ff_data['images_ff'], # (N,H,W,3)
        }
        return scene 

    def __len__(self):
        return 1
    


