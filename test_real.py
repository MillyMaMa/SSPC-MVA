import torch 
import numpy as np
import sys
import os
from data.RealDataset import RealDataset
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from models.mva import MVA
from loss.loss import  distChamfer, directed_hausdorff   
from tqdm import tqdm

from utils.Logger import Logger
from utils.output_xyz import output_xyz
from torch_scatter import scatter_min


def resample(pc, N):
    ind = torch.randint(0,pc.shape[1],(N,))
    return pc[:,ind]


def synthesize(pcs_input,R=None):
    pcs = torch.transpose(pcs_input,2,3)
    pcs = torch.matmul(pcs,R)
    res = 16
    inds = torch.clamp(torch.round(res*(pcs+0.5)),0,res).long()
    inds = (res*inds[:,:,:,1]+inds[:,:,:,2]).long()        
    new_pcs = []
    cnt = []
    for i in range(pcs.shape[0]):
        new_pcs_i = []
        cnt_i = []
        for j in range(pcs.shape[1]):
            pc = pcs[i,j]
            ind = inds[i,j]
            un, inv = torch.unique(ind,return_inverse=True)
            out, argmin = scatter_min(pc[:,0], inv, dim=0)
            new_pc = pcs_input[i,0][:,argmin]
            new_pc = resample(new_pc, 1024).unsqueeze(0)  
            new_pcs_i.append(new_pc)
            cnt_i.append(argmin.shape[0])
        new_pcs_i = torch.cat(new_pcs_i,0).unsqueeze(0)
        cnt.append(cnt_i)
        new_pcs.append(new_pcs_i)
    new_pcs = torch.cat(new_pcs,0)
    return(new_pcs), cnt


class Tester(object):

    def __init__(self, args):
        self.args =args
        self.batch_size = 1
        self.test_dataset = RealDataset(args, args.class_name, 'test')
        self.test_loader = torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size= self.batch_size,  
                shuffle=False,
                num_workers= 1
            )

        self.num_points = args.num_points                           
        self.device = torch.device("cuda")
        self.model = MVA()
        self.haus = directed_hausdorff
        self.eval_CD = distChamfer
        self.num_syn = 8
        sys.stdout = Logger(os.path.join(self.args.output_dir, 'log.txt'))

    def test_epoch(self):
        self.model.eval()
        with torch.no_grad():
            UCD_buf = []
            UHD_buf = []  
            num_inst = 0
            for iter, data in enumerate(tqdm(self.test_loader)):
                
                inputs, index, R = data  
                inputs = inputs.to(self.device) 
                R = R.to(self.device)            
                inputs = torch.transpose(inputs, 1,2)   

                output2 = self.model(inputs) 

                syns, cnt_syn = synthesize(inputs.unsqueeze(1),R)  

                inst = 0
                for n in range(self.batch_size):
                    inst = num_inst + n
                    output_xyz(torch.transpose(output2, 1,2)[n], self.args.output_dir + '/cpc/'+str(inst)+'.ply')
                    output_xyz(torch.transpose(inputs, 1,2)[n] , self.args.output_dir +  '/ipc/'+str(inst)+'.ply')

                    if inst==0: 
                        for m in range(self.num_syn):
                            output_xyz(torch.transpose(syns[:,m], 1,2)[n] , self.args.output_dir+  '/v/'+str(inst)+ '_'+str(m) +'.ply')
                num_inst = (iter+1)*self.batch_size

                dist1, dist2, _,_ = self.eval_CD(inputs.transpose(1,2), output2.transpose(1,2))
                UCD = dist1.mean()
                UHD = self.haus(inputs,output2)

                UCD_buf.append(UCD.detach().cpu().numpy())
                UHD_buf.append(UHD.detach().cpu().numpy())
            
            print(f'[Test] UCD: %.8f  UHD: %.5f' % (np.mean(UCD_buf), np.mean(UHD_buf)))


    def run(self):

        print(f"Testing start~")
        print(f'weight dir: {self.args.weight}')
        print(f'test dir: {self.args.output_dir}')
        print(f'exp_name: {self.args.experiment_id} dataset: {self.args.dataset_name} class_name: {self.args.class_name}  batch_size: {self.batch_size} dataset dir: {self.args.root}' )

        self.model=self.model.to(self.device)
        assert self.args.weight!=None, 'Error: args.weight=None'
        self.model.load_state_dict(torch.load(self.args.weight))
        self.test_epoch() 
        print(f"Test finish.")
        






     







    

