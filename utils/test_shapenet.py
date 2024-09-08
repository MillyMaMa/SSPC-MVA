import torch 
import sys
import os
from data.data import NetDataset 
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from models.pcn import PCN
from loss.loss import chamfer_distance_kdtree
from tqdm import tqdm
from utils.Logger import Logger
import time
from utils.output_xyz import output_xyz


class Tester(object):

    def __init__(self, args):
        self.test_dataset = NetDataset(args, args.root, split='test')  
        self.batch_size = 32
        self.test_loader = torch.utils.data.DataLoader( self.test_dataset, batch_size= self.batch_size, shuffle=False, num_workers= 1) 
        self.num_points = args.num_points                           
        self.device = torch.device("cuda")
        self.model = PCN()
        self.loss1 = chamfer_distance_kdtree  
        self.output_dir = args.output_dir
        self.dataset_name = args.dataset_name
        sys.stdout = Logger(os.path.join(self.output_dir, 'log.txt'))
        self.args =args

    def test_epoch(self):
        self.model.eval()
        with torch.no_grad():
            chamf = []
            Dist1 = []
            Dist2 = []
            OUT_o2, OUT_gt, OUT_pc = [], [], []

            for iter, test_data in enumerate(tqdm(self.test_loader)):
                test_pc, test_gt, test_R = test_data  
                test_pc = test_pc.to(self.device)
                test_gt = test_gt.to(self.device)
                test_pc = torch.transpose(test_pc, 2,3) 
                test_gt = torch.transpose(test_gt, 2,3) 
                out_o2 = []
                for v in range(test_pc.shape[1]): 

                    total = sum([param.nelement() for param in self.model.parameters()])
                    print("Number of parameter: %.2fM" % (total/1e6))
                    start = time.time()
                    test_output2= self.model(test_pc[:,v]) 
                    torch.cuda.synchronize()
                    end = time.time()
                    print('infer_time:', end-start)

                    out_o2.append(torch.transpose(test_output2, 1,2).unsqueeze(1))  
                    dist1, dist2 = self.loss1(test_gt[:,v], test_output2)    
                    Dist1.append(torch.mean(dist1).cpu().numpy())
                    Dist2.append(torch.mean(dist2).cpu().numpy())
                    chamf.append((torch.mean(dist1)+torch.mean(dist2)).cpu().numpy())
                out_o2 = torch.cat(out_o2,1)
                OUT_o2.append(out_o2)

                OUT_gt.append(test_gt) 
                OUT_pc.append(test_pc)    
            
            OUT_o2 = torch.cat(OUT_o2,0)

            OUT_gt = torch.cat(OUT_gt,0)  
            OUT_pc = torch.cat(OUT_pc,0)
            OUT_gt = torch.transpose(OUT_gt,2,3)
            OUT_pc = torch.transpose(OUT_pc,2,3)
            for k in range(OUT_gt.shape[0]): 
                output_xyz(OUT_gt[k,0] , self.output_dir +  '/gt/'+str(k)+'.ply')
                for v in range(OUT_pc.shape[1]):
                    output_xyz(OUT_pc[k,v] , self.output_dir +  '/ipc/'+str(v)+'/'+str(k)+'.ply')          
                            
            Dist1 = sum(Dist1)/len(Dist1)  
            Dist2 = sum(Dist2)/len(Dist2)  
            chamf = sum(chamf)/len(chamf) 

            for v in range(OUT_o2.shape[1]):
                for k in range(OUT_o2.shape[0]):
                    output_xyz(OUT_o2[k,v] , self.output_dir +  '/cpc/'+str(v)+'/'+str(k)+'.ply')

            print(f'[Test] coverage: %.5f  precision: %.5f  chamf: %.5f' % (Dist1, Dist2, chamf))
            torch.cuda.synchronize()

    def run(self):
        print(f"Testing start~")
        print(f'weight dir: {self.args.weight}')
        print(f'test dir: {self.output_dir}')
        print(f'exp_name: {self.args.experiment_id} dataset: {self.args.dataset_name} class_name: {self.args.class_name} batch_size: {self.batch_size} dataset dir: {self.args.root}' )

        self.model=self.model.to(self.device)
        assert self.args.weight!=None, 'Error: args.weight=None'
        self.model.load_state_dict(torch.load(self.args.weight))

        self.test_epoch() 
        print(f"Test finish.")








 
 

