import torch
import torch.nn as nn
from geomloss import SamplesLoss
import math
import ot
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

_EPS = 1e-8


class OTLoss(nn.Module):
    
    def __init__(self, args=None, device='cuda:0'):
        super(OTLoss, self).__init__()
        self.device = torch.device(device if isinstance(device, str) else str(device))

        self.blur = 0.05
        self.max_points = int(getattr(args, 'ot_max_points', 1024)) if args else 1024
        self.sigma = 1.0


        self.sinkhorn_loss = SamplesLoss(
            loss="sinkhorn", 
            p=2, 
            blur=self.blur,
            backend="tensorized"
        )
   

        self.plan_loss = SamplesLoss(
            loss='sinkhorn', 
            p=2, 
            debias=True,
            blur=0.1 ** 0.5,  # Line 120: blur=0.1**(1/2)
            backend='tensorized'
        )
        
    
    def _safe_to_float32(self, t):
        return t.float() if t.dtype != torch.float32 else t
    
    def _sample_nodes(self, N1, N2, seed=None):

        N = min(N1, N2)
        n_pts = min(N, self.max_points)
        
        if n_pts <= 0:
            return 0, None, None
        

        need_sampling = N > self.max_points
        
        if seed is not None and need_sampling:

            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
            idx = torch.randperm(N1, generator=generator, device=self.device)[:n_pts]
        else:

            idx = torch.randperm(N1, device=self.device)[:n_pts]
    
        return n_pts, idx, idx
    
    def compute_loss(self, view1_data, view2_data, return_plans=False, sigma=None):

        if sigma is None:
            sigma = self.sigma
            

        x1, z1, edge_index1 = view1_data
        x2, z2, edge_index2 = view2_data
        
        x1, x2 = x1.to(self.device), x2.to(self.device)
        z1, z2 = z1.to(self.device), z2.to(self.device)
        
        N1, N2 = x1.shape[0], x2.shape[0]
        

        n_pts, idx_x, idx_y = self._sample_nodes(N1, N2, seed=3407)
        if n_pts == 0:
            zero = torch.tensor(0.0, device=self.device)
            if return_plans:
                return zero, zero, None, None
            return zero, zero
        
        x1_sub, x2_sub = x1[idx_x], x2[idx_y]
        z1_sub, z2_sub = z1[idx_x], z2[idx_y]
        

        Mp = torch.cdist(x1_sub, x2_sub, p=2)
        Mb = torch.cdist(z1_sub, z2_sub, p=2)
        

        if sigma < 1.0:

            adj1 = to_dense_adj(edge_index1, max_num_nodes=N1).squeeze(0)
            adj2 = to_dense_adj(edge_index2, max_num_nodes=N2).squeeze(0)
            C1 = adj1[idx_x][:, idx_x].float()
            C2 = adj2[idx_y][:, idx_y].float()


            h1 = ot.unif(n_pts, type_as=x1_sub)
            

            P = ot.gromov.semirelaxed_fused_gromov_wasserstein(
                M=Mp, C1=C1, C2=C2, p=h1, 
                symmetric=True, alpha=(1.0 - sigma), log=False
            )
            

            C1_sq, C2_sq = C1**2, C2**2
            mu = P.sum(1, keepdim=True)
            nu = P.sum(0, keepdim=True)
            
            constC1 = torch.matmul(C1_sq, mu)
            constC2 = torch.matmul(nu, C2_sq.t())
            Mp2 = constC1 + constC2 - 2 * torch.matmul(torch.matmul(C1, P), C2.t())
            

            Mp2 = F.normalize(Mp2)
            Mp = sigma * Mp + (1 - sigma) * Mp2
            

            P = P / (P.sum(dim=-1, keepdim=True) + _EPS)

        else:

            self.plan_loss.potentials = True
            

            u_p, v_p = self.plan_loss(
                self._safe_to_float32(x1_sub), 
                self._safe_to_float32(x2_sub)
            )
            P = torch.exp((u_p.unsqueeze(1) + v_p.unsqueeze(0) - Mp) / 0.1)
            P = P / (P.sum(dim=-1, keepdim=True) + _EPS)
        

        self.plan_loss.potentials = True
        u_b, v_b = self.plan_loss(
            self._safe_to_float32(z1_sub), 
            self._safe_to_float32(z2_sub)
        )
        B = torch.exp((u_b.unsqueeze(1) + v_b.unsqueeze(0) - Mb) / 0.1)
        B = B / (B.sum(dim=-1, keepdim=True) + _EPS)
        

        scale = math.sqrt(n_pts / 1024.0)
        
        struct_loss = self.sinkhorn_loss(
            self._safe_to_float32(Mp), 
            self._safe_to_float32(Mb)
        ) / scale
        

        match_loss = torch.linalg.matrix_norm(P - B, ord='fro') / scale  # 归一化
        
        if return_plans:
            return struct_loss, match_loss, P, B
        return struct_loss, match_loss


def rowwise_ce_distill(student_plan, teacher_plan, eps=1e-8, scale=1.0):

    student_plan = torch.clamp(student_plan, min=eps)
    teacher_plan = torch.clamp(teacher_plan, min=eps)
    

    ce = -(teacher_plan * torch.log(student_plan)).sum(dim=-1)

    return ce.mean() / scale