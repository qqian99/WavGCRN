import torch.optim as optim
import math
import numpy as np
from netp import *
import util
import time
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid


class Trainer():
    def __init__(self,
                 adj1,
                 adj2,
                 model,
                 lrate,
                 wdecay,
                 clip,
                 step_size,
                 seq_out_len,
                 scaler,
                 device,
                 cl=True,
                 new_training_method=False):
        self.adj1 = adj1
        self.adj2 = adj2
        
        self.scaler = scaler
        self.model = model
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=lrate,
                                    weight_decay=wdecay)
        self.loss = util.hy_mae#masked_mae
        self.clip = clip
        self.step = step_size
        self.device = device
        self.iter = 0
        self.task_level = 1
        self.seq_out_len = seq_out_len
        self.cl = cl
        self.new_training_method = new_training_method

    def notears_linear(self, X, lambda1, loss_type, max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.0001):
        """Solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0 using augmented Lagrangian.

        Args:
            X (np.ndarray): [n, d] sample matrix
            lambda1 (float): l1 penalty parameter
            loss_type (str): l2, logistic, poisson
            max_iter (int): max num of dual ascent steps
            h_tol (float): exit if |h(w_est)| <= htol
            rho_max (float): exit if rho >= rho_max
            w_threshold (float): drop edge if |weight| < threshold

        Returns:
            W_est (np.ndarray): [d, d] estimated DAG
        """

        def _loss(W):
            """Evaluate value and gradient of loss."""
            M = X @ W
            if loss_type == 'l2':
                R = X - M
                loss = 0.5 / X.shape[0] * (R ** 2).sum()
                G_loss = - 1.0 / X.shape[0] * X.T @ R
            elif loss_type == 'logistic':
                loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
                G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
            elif loss_type == 'poisson':
                S = np.exp(M)
                loss = 1.0 / X.shape[0] * (S - X * M).sum()
                G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
            else:
                raise ValueError('unknown loss type')
            return loss, G_loss

        def _h(W):
            """Evaluate value and gradient of acyclicity constraint."""
            E = slin.expm(W * W)  # (Zheng et al. 2018)
            h = np.trace(W)
            #     # A different formulation, slightly faster at the cost of numerical stability
            #     M = np.eye(d) + W * W / d  # (Yu et al. 2019)
            #     E = np.linalg.matrix_power(M, d - 1)
            #     h = (E.T * M).sum() - d
            G_h = E.T * W * 2
            return h, G_h

        def _adj(w):
            """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
            return (w[:d * d] - w[d * d:]).reshape([d, d])

        def _func(w):
            """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
            W = _adj(w)
            loss, G_loss = _loss(W)
            h, G_h = _h(W)
            obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
            G_smooth = G_loss + (rho * h + alpha) * G_h
            g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), axis=None)
            return obj, g_obj

        n, d = X.shape
        w_est, rho, alpha, h = np.random.rand(2 * d * d), 1.0, 0.0, np.inf  # double w_est into (w_pos, w_neg)
        bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
        if loss_type == 'l2':
            X = X - np.mean(X, axis=0, keepdims=True)

        for _ in range(max_iter):
            w_new, h_new = None, None
            while rho < rho_max:
                sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
                w_new = sol.x
                #print("w_new: ", w_new.sum())
                h_new, _ = _h(_adj(w_new))
                if h_new > 0.25 * h:
                    rho *= 10
                else:
                    break
            #print("change: ", (w_new-w_est).sum())
            w_est, h = w_new, h_new
            alpha += rho * h
            if h <= h_tol or rho >= rho_max:
                break
        W_est = _adj(w_est)
        print("learned graph: ", W_est.sum(), np.max(W_est))
        #W_est[np.abs(W_est) < w_threshold] = 0
        #print(W_est)
        #print(W_est.sum())
        return torch.from_numpy(W_est).to(self.device)

    def train(self, input, real_val, ycl, idx=None, batches_seen=None):
        self.iter += 1

        if self.iter % self.step == 0 and self.task_level < self.seq_out_len:
            self.task_level += 1
            if self.new_training_method:
                self.iter = 0

        #print("In trainer2: ", self.task_level)
        self.model.train()
        self.optimizer.zero_grad()
        if self.cl:
            output, md1, md2, embedding_feature1, embedding_feature2 = self.model(input, md1=self.adj1, md2=self.adj2,
                                idx=idx,
                                ycl=ycl,
                                batches_seen=self.iter,
                                task_level=self.task_level)
        else:
            output, md1, md2, embedding_feature1, embedding_feature2 = self.model(input, md1=self.adj1, md2=self.adj2,
                                idx=idx,
                                ycl=ycl,
                                batches_seen=self.iter,
                                task_level=self.seq_out_len)

        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)

        if self.cl:

            loss = self.loss(predict[:, :, :, :self.task_level],
                             real[:, :, :, :self.task_level], md1, md2, embedding_feature1, embedding_feature2, 0.0)
            mape = util.masked_mape(predict[:, :, :, :self.task_level],
                                    real[:, :, :, :self.task_level],
                                    0.0).item()
            rmse = util.masked_rmse(predict[:, :, :, :self.task_level],
                                    real[:, :, :, :self.task_level],
                                    0.0).item()
        else:
            loss = self.loss(predict, real, md1, md2, embedding_feature1, embedding_feature2, 0.0)
            mape = util.masked_mape(predict, real, 0.0).item()
            rmse = util.masked_rmse(predict, real, 0.0).item()

        loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        self.optimizer.step()

        return loss.item(), mape, rmse

    def graph_learning(self, input, real_val, ycl, idx=None, batches_seen=None):

        if self.cl:
            output, md1, md2, embedding_feature1, embedding_feature2 = self.model(input, md1=self.adj1, md2=self.adj2,
                                idx=idx,
                                ycl=ycl,
                                batches_seen=self.iter,
                                task_level=self.task_level)
            t1 = time.time()
            print("what about input1: ", (embedding_feature1-np.mean(embedding_feature1)).sum())
            self.adj1 = self.notears_linear(X=embedding_feature1, lambda1=0.1, loss_type='l2').unsqueeze(0).repeat(64, 1, 1).float()
            #if self.adj1.sum() < 0.00001:
                #print("repeat")
                #b=self.notears_linear(X=embedding_feature1, lambda1=0.1, loss_type='l2').unsqueeze(0).repeat(64, 1, 1)
                #print("add uniform")
                #embedding_feature1 += np.random.uniform(0.0, 1e-11, (64,207))
                #a=self.notears_linear(X=embedding_feature1, lambda1=0.1, loss_type='l2').unsqueeze(0).repeat(64, 1, 1).float()
                #print("pure uniform")
                #embedding_feature1 = np.random.uniform(0.0, 1e-11, (64,207))
                #c=self.notears_linear(X=embedding_feature1, lambda1=0.1, loss_type='l2').unsqueeze(0).repeat(64, 1, 1).float()
            t2 = time.time()
            print("first graph learning time cost: ", t2-t1)
            print("what about input2: ", (embedding_feature2-np.mean(embedding_feature2)).sum())
            t1 = time.time()
            self.adj2 = self.notears_linear(X=embedding_feature2, lambda1=0.1, loss_type='l2').unsqueeze(0).repeat(64, 1, 1).float()
            #print("add uniform2")
            #embedding_feature2 += np.random.uniform(0.0, 1e-11, (64,207))
            #a=self.notears_linear(X=embedding_feature1, lambda1=0.1, loss_type='l2').unsqueeze(0).repeat(64, 1, 1).float()
            t2 = time.time()
            print("second graph learning time cost: ", t2-t1)
            
        else:
            output, md1, md2, embedding_feature1, embedding_feature2 = self.model(input, md1=self.adj1, md2=self.adj2,
                                idx=idx,
                                ycl=ycl,
                                batches_seen=self.iter,
                                task_level=self.seq_out_len)
            self.adj1 = self.notears_linear(X=embedding_feature1, lambda1=0.1, loss_type='l2').unsqueeze(0).repeat(64, 1, 1).float()
            self.adj2 = self.notears_linear(X=embedding_feature2, lambda1=0.1, loss_type='l2').unsqueeze(0).repeat(64, 1, 1).float()

        return self.adj1, self.adj2

    def eval(self, input, real_val, ycl):
        self.model.eval()
        with torch.no_grad():
            output, md1, md2, embedding_feature1, embedding_feature2 = self.model(input, md1=self.adj1, md2=self.adj2, ycl=ycl)
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, md1, md2, embedding_feature1, embedding_feature2, 0.0)
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mape, rmse
