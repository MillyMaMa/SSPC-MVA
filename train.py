import torch 
import numpy as np
import sys
import os
from data.data import NetDataset 
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from models.pcn import PCN
from loss.loss import chamfer_distance_kdtree
from loss.loss import MSELoss 
import torch.optim as optim
from tqdm import tqdm
import time
from utils.find_Nearest_Neighbor import find_NN_batch
from utils.Logger import Logger

import time
from utils.output_xyz import output_xyz

from torch_scatter import scatter_min
import open3d as o3d
from utils.misc import AverageMeter


def resample(pc, N=2500):
    ind = torch.randint(0,pc.shape[1],(N,))
    return pc[:,ind]
    
def downsample(point,num):
   outs = []
   for i in range(point.shape[0]):
       ind = torch.randint(0,point.shape[2],(num,))
       out = point[i,:,ind].unsqueeze(0)
       outs.append(out)
   return torch.cat(outs,0)

def read_pcd(filename):
    pcd = o3d.io.read_point_cloud(filename)
    return np.array(pcd.points)



def synthesize(pcs_input,R=None):
    pcs = torch.transpose(pcs_input,2,3)
    pcs = torch.matmul(pcs,R)
    res = 64
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
            new_pc = resample(new_pc).unsqueeze(0) 
            new_pcs_i.append(new_pc)
            cnt_i.append(argmin.shape[0])
        new_pcs_i = torch.cat(new_pcs_i,0).unsqueeze(0)
        cnt.append(cnt_i)
        new_pcs.append(new_pcs_i)
    new_pcs = torch.cat(new_pcs,0)
    return(new_pcs), cnt


def sample_spherical(npoints, ndim=3):
    x = np.linspace(-0.5,0.5,npoints)
    y = np.linspace(-0.5,0.5,npoints)
    z = np.linspace(-0.5,0.5,npoints)
    X,Y,Z = np.meshgrid(x,y,z)
    vec = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
    return vec
    
def grids(N=128):
    grid_sphere = torch.Tensor(sample_spherical(N)).unsqueeze(0).view(1,3,-1)

    

class trainer(object):

    def __init__(self, args):
        self.train_dataset = NetDataset(args, args.root, split='train')
        self.test_dataset = NetDataset(args, args.root, split='test')  
        self.batch_size = args.batch_size
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, 
            batch_size= self.batch_size, 
            shuffle=True, 
            num_workers= 1
            )
        self.test_loader = torch.utils.data.DataLoader( 
            self.test_dataset, 
            batch_size= 2, 
            shuffle=False, 
            num_workers= 1
            )
        self.num_points = args.num_points                           
        self.device = torch.device("cuda")
        self.model = PCN()
        self.find_NN_batch = find_NN_batch().to(self.device)  
        self.parameter1 = self.model.parameters()
        self.optimizer1 = optim.Adam(self.parameter1, lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-6)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer1, step_size=200, gamma=0.5)
        self.gamma = 0.5
        self.loss1 = chamfer_distance_kdtree 
        self.loss2 = MSELoss()    
        self.epochs = args.epochs
        self.snapshot_interval = args.snapshot_interval
        self.experiment_id = args.experiment_id

        self.snapshot_root = args.snapshot_dir + '%s' % self.experiment_id
        self.save_dir = os.path.join(self.snapshot_root, 'models/')
        if os.path.isdir(self.save_dir) is False:
            os.makedirs(self.save_dir)

        self.dataset_name = args.dataset_name
        self.num_syn = 8
        sys.stdout = Logger(os.path.join(self.snapshot_root, 'log.txt'))
        self.args =args

        self.losses = AverageMeter(["loss", "loss1", "loss2"])
        self.test_losses = AverageMeter(["loss", "loss1", "loss2"])


    def _snapshot(self, epoch):
        state_dict = self.model.state_dict()
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for key, val in state_dict.items():
            if key[:6] == 'module':
                name = key[7:]  
            else:
                name = key
            new_state_dict[name] = val
            
        save_dir = os.path.join(self.save_dir, self.dataset_name)
        torch.save(new_state_dict, save_dir + "_" + str(epoch) + '.pkl')


    def train_epoch(self, epoch):

        loss_buf=[]
        loss1_buf=[]
        loss2_buf=[]
        self.model=self.model.to(self.device)
        
        if self.args.resume!=None:
           self.model.load_state_dict(torch.load(self.args.resume))
           self.model.train()

        for iter, data in enumerate(tqdm(self.train_loader)):
            inputs, gt, R = data  
            inputs = inputs.to(self.device)
            gt = gt.to(self.device)
            R = R.to(self.device)
            inputs = torch.transpose(inputs, 1,2)
            gt = torch.transpose(gt, 1,2)   
            self.optimizer1.zero_grad()
            output2= self.model(inputs)

            dist21, dist22 = self.loss1(inputs, output2)  
            loss1 = 0.9*torch.mean(dist21)+0.1*torch.mean(dist22)      
            loss2 = 0
            out_final = [output2]
            syns, cnt_syn = synthesize(inputs.unsqueeze(1),R) 

            for i in range(self.num_syn):
                o_syn2 = self.model(syns[:,i].detach())
                out_final.append(o_syn2)
                loss2 += self.loss2(o_syn2,output2)/self.num_syn 

            out_9 = out_final
            out_final = sum(out_final)/(self.num_syn+1)  

            if epoch %1 ==0 and iter<40:
                output_xyz(torch.transpose(gt, 1,2)[0] , self.args.output_dir+  '/train/gt_epoch'+str(epoch+1)+'_iter'+str(iter+1)+'.ply')
                output_xyz(torch.transpose(inputs, 1,2)[0] , self.args.output_dir+  '/train/ipc_epoch'+str(epoch+1)+'_iter'+str(iter+1)+'.ply')
                output_xyz(torch.transpose(syns[:,0], 1,2)[0] , self.args.output_dir+  '/train/v0_epoch'+str(epoch+1)+'_iter'+str(iter+1)+'.ply')
                output_xyz(torch.transpose(syns[:,1], 1,2)[0] , self.args.output_dir+  '/train/v1_epoch'+str(epoch+1)+'_iter'+str(iter+1)+'.ply')
                output_xyz(torch.transpose(syns[:,2], 1,2)[0] , self.args.output_dir+  '/train/v2_epoch'+str(epoch+1)+'_iter'+str(iter+1)+'.ply')
                output_xyz(torch.transpose(syns[:,3], 1,2)[0] , self.args.output_dir+  '/train/v3_epoch'+str(epoch+1)+'_iter'+str(iter+1)+'.ply')
                output_xyz(torch.transpose(syns[:,4], 1,2)[0] , self.args.output_dir+  '/train/v4_epoch'+str(epoch+1)+'_iter'+str(iter+1)+'.ply')
                output_xyz(torch.transpose(syns[:,5], 1,2)[0] , self.args.output_dir+  '/train/v5_epoch'+str(epoch+1)+'_iter'+str(iter+1)+'.ply')
                output_xyz(torch.transpose(syns[:,6], 1,2)[0] , self.args.output_dir+  '/train/v6_epoch'+str(epoch+1)+'_iter'+str(iter+1)+'.ply')
                output_xyz(torch.transpose(syns[:,7], 1,2)[0] , self.args.output_dir+  '/train/v7_epoch'+str(epoch+1)+'_iter'+str(iter+1)+'.ply')

 
            dist3_1, dist3_2 = self.loss1(out_final, gt) 
            trainset_eval = torch.mean(dist3_1+dist3_2).detach().cpu() 

            loss = (loss1+10*loss2) 
            loss.backward()
            self.optimizer1.step()
            loss1_buf.append(loss1.detach().cpu().numpy())
            loss2_buf.append(loss2.detach().cpu().numpy()) 
            loss_buf.append(loss.detach().cpu().numpy())
            del loss
        
        self.scheduler.step()

        loss1_mean = np.mean(loss1_buf)
        loss2_mean = np.mean(loss2_buf)
        loss_mean = np.mean(loss_buf)
        trainset_eval_mean = torch.mean(trainset_eval)

        if epoch%1 ==0:
            self.train_hist['loss'].append(loss_mean)

        print(f'[Epoch %d] loss %.5f  loss1(i-o) %.5f  loss2(8-o) %.5f  trainset_eval %.5f' % ((epoch+1), loss_mean, loss1_mean, loss2_mean, trainset_eval_mean))
        
        return loss_mean  


    def test_epoch(self, epoch, args):
        test_start = torch.cuda.Event(enable_timing=True)    
        test_end = torch.cuda.Event(enable_timing = True)
        test_start.record()
        
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

                    test_output2= self.model(test_pc[:,v])

                    out_o2.append(torch.transpose(test_output2, 1,2).unsqueeze(1))  
                    dist1, dist2 = self.loss1(test_gt[:,v], test_output2)    
                    Dist1.append(torch.mean(dist1).cpu().numpy())
                    Dist2.append(torch.mean(dist2).cpu().numpy())
                    chamf.append((torch.mean(dist1)+torch.mean(dist2)).cpu().numpy())
                    
                out_o2 = torch.cat(out_o2,1)
                OUT_o2.append(out_o2)
                if epoch==0:
                   OUT_gt.append(test_gt) 
                   OUT_pc.append(test_pc)    
            
            OUT_o2 = torch.cat(OUT_o2,0)
            if epoch==0:
               OUT_gt = torch.cat(OUT_gt,0)
               OUT_pc = torch.cat(OUT_pc,0)
               OUT_gt = torch.transpose(OUT_gt,2,3)
               OUT_pc = torch.transpose(OUT_pc,2,3)
               for k in range(OUT_gt.shape[0]):
                   output_xyz(OUT_gt[k,0] , args.output_dir+  '/test/gt/'+str(k)+'.ply')
                   for v in range(OUT_pc.shape[1]):
                       output_xyz(OUT_pc[k,v] , args.output_dir+  '/test/ipc/'+str(v)+'/'+str(k)+'.ply')
                               
                            
            Dist1 = sum(Dist1)/len(Dist1)  
            Dist2 = sum(Dist2)/len(Dist2)  
            chamf = sum(chamf)/len(chamf) 

            if epoch > (self.epochs-50):
               np.save(args.output_dir+ '/result.npy',np.array([chamf,Dist1,Dist2,epoch]))

            print(f'[TestEpoch %d] coverage: %.5f  precision: %.5f  chamf: %.5f' % ((epoch+1), Dist1, Dist2, chamf))
            test_end.record()  
            torch.cuda.synchronize()

        return (Dist1+Dist2), Dist1, Dist2, chamf
    


    def run(self,args):
        self.train_hist = { 'loss': [], 'per_epoch_time': [], 'total_time': []}

        best_loss = 10000
        test_loss = 10000
        print(f"Training start~")
        print(f'exp_name: {self.args.experiment_id} dataset: {self.args.dataset_name} class_name: {self.args.class_name} batch_size: {self.args.batch_size} epoch: {self.args.epochs} initial_Lr: {self.args.lr} scheduler_gamma: {self.gamma}' )
        print(f'snapshot dir: {self.snapshot_root} dataset dir: {self.args.root} test interval: {self.args.snapshot_interval}')

        start_time = time.time()
        
        epoch_init = 0
        total_time = 0    
        for epoch in tqdm(range(epoch_init, self.epochs)):  
            start_epoch = torch.cuda.Event(enable_timing=True)
            end_epoch = torch.cuda.Event(enable_timing = True)
            start_epoch.record()
  
            loss = self.train_epoch(epoch)  

            if (epoch+1) % 1==0: 
                test_start_time = time.time()
                test_loss, test_coverage, test_precision, test_chamf = self.test_epoch(epoch, args) 
                test_end_time = time.time()
                print(f"[TestEpoch %d] TestTime = %.f (s)" % ((epoch+1), test_end_time - test_start_time))
                if test_loss < best_loss:
                    best_loss = test_loss
                    self._snapshot('best_test_loss')  
                    print(f"[TestEpoch %d] best loss: %.5f , Save model: best_test_loss.pkl" % ((epoch+1), best_loss))
            

            end_epoch.record()
            torch.cuda.synchronize()
            epoch_time = start_epoch.elapsed_time(end_epoch)/1000
            total_time += epoch_time
            print(f'epoch_time: '+str("%.1f" % epoch_time)+' s')

        end_time = time.time()
        self.train_hist['total_time'].append(end_time- start_time)

        print(f"Total [%d] epochs time: %.1f hours" % (self.epochs, self.train_hist['total_time'][0]/(3600)))
        print(f"Training finish!")








 
 

