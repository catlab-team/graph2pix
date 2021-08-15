import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset, make_mv_dataset, make_sv_dataset, make_art_dataset
from PIL import Image
import sys

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

        ### input A (label maps)
        dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        if opt.mv:
            self.A_paths = make_mv_dataset(self.dir_A)
        else:
            self.A_paths = make_sv_dataset(self.dir_A)

        ### input B (real images)
        if opt.isTrain or opt.use_encoded_image:
            dir_B = '_B' if self.opt.label_nc == 0 else '_img'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)  
            self.B_paths = make_dataset(self.dir_B)

        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:                              
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        self.dataset_size = len(self.A_paths)
      
    def __getitem__(self, index):    
        ### input A (label maps)
        A_path = self.A_paths[index]
        if self.opt.mv:
            A = []
            for path in A_path:
                A.append(Image.open(path))
            params = get_params(self.opt, A[0].size)
            if self.opt.label_nc == 0:
                transform_A = get_transform(self.opt, params)
                A_tensor = []
                for img in A:
                    A_tensor.append(transform_A(img.convert('RGB')))
            else:
                transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
                A_tensor = []
                for img in A:
                    A_tensor.append(transform_A(img) * 255.0)
        else:
            A = Image.open(A_path)  
            params = get_params(self.opt, A.size)
            if self.opt.label_nc == 0:
                transform_A = get_transform(self.opt, params)
                A_tensor = transform_A(A.convert('RGB'))
            else:
                transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
                A_tensor = transform_A(A) * 255.0
        
        
        B_tensor = inst_tensor = feat_tensor = 0
        ### input B (real images)
        if self.opt.isTrain or self.opt.use_encoded_image:
            B_path = self.B_paths[index]   
            B = Image.open(B_path).convert('RGB')
            transform_B = get_transform(self.opt, params)      
            B_tensor = transform_B(B)
        
        ### if using instance maps        
        if not self.opt.no_instance:
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = transform_A(inst)

            if self.opt.load_features:
                feat_path = self.feat_paths[index]            
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform_A(feat))  

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 
                      'feat': feat_tensor, 'path': A_path}

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'
    
    
class ArtDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    
        data_dir = 'train_art' if opt.phase == 'train' else 'test_art'
        data_dir = self.root + data_dir
        self.label_paths, self.real_paths = make_art_dataset(data_dir, self.opt.mv)

        ### load precomputed instance-wise encoded features

        self.dataset_size = len(self.label_paths)
      
    def __getitem__(self, index):    
        ### label maps
        label_pth = self.label_paths[index]
        if self.opt.mv:
            label = []
            for path in label_pth:
                # print(path)
                label.append(Image.open(path))
            params = get_params(self.opt, label[0].size)
            if self.opt.label_nc == 0:
                transform_label = get_transform(self.opt, params)
                label_tensor = []
                for img in label:
                    label_tensor.append(transform_label(img.convert('RGB')))
            else:
                transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
                label_tensor = []
                for img in label:
                    label_tensor.append(transform_label(img) * 255.0)
        else:
            label = Image.open(label_pth)  
            params = get_params(self.opt, label.size)
            if self.opt.label_nc == 0:
                transform_label = get_transform(self.opt, params)
                label_tensor = transform_label(label.convert('RGB'))
            else:
                transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
                label_tensor = transform_label(label) * 255.0
        
        
        real_tensor = inst_tensor = feat_tensor = 0
        ### real images
        if self.opt.isTrain or self.opt.use_encoded_image:
            real_pth = self.real_paths[index]
            real = Image.open(real_pth).convert('RGB')
            transform_real = get_transform(self.opt, params)      
            real_tensor = transform_real(real)
        
        ### if using instance maps        
        if not self.opt.no_instance:
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = transform_label(inst)

            if self.opt.load_features:
                feat_path = self.feat_paths[index]            
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform_label(feat))  

        input_dict = {'label': label_tensor, 'inst': inst_tensor, 'image': real_tensor, 
                      'feat': feat_tensor, 'path': label_pth}

        return input_dict

    def __len__(self):
        return len(self.label_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'ArtDataset'