"""
Original code from:
https://github.com/marcvornehm/M-DIP/blob/main/dip/lps.py

Put all necessary code for the L+S baseline here, 
and adjusted it to fit the data and setup used in this project.
"""
import torch
import torch.types
from tqdm import tqdm
from pathlib import Path

from cardiac_diffusion.mri import fftnc, ifftnc


class LowRankPlusSparse:
    def __init__(self, d: torch.Tensor, C: torch.Tensor, U: torch.Tensor, intermediate_path: Path, intermediate_save_frequency: int) -> None:
        """L+S reconstruction (Otazo et al. https://doi.org/10.1002/mrm.25240)

        Args:
            d (torch.Tensor): k-space measurements [t, c, x, y, (2)]
            C (torch.Tensor): Coil sensitivity maps [c, x, y, (2)]
            U (torch.Tensor): Undersampling mask [t, c, x, y]
        """
        if not d.is_complex() and d.shape[-1] == 2:
            d = torch.view_as_complex(d.contiguous())
        if not C.is_complex() and C.shape[-1] == 2:
            C = torch.view_as_complex(C.contiguous())
        C = C[None]  # [t=1, c, x, y]
        self.d = d.reshape(*d.shape[:2], -1).permute([1, 2, 0])  # [c, x*y, t]
        self.C = C.reshape(*C.shape[:2], -1).permute([1, 2, 0])  # [c, x*y, t=1]
        self.U = U.reshape(*U.shape[:2], -1).permute([1, 2, 0])  # [c, x*y, t]
        self.image_size = d.shape[2:4]

        self.intermediate_path = intermediate_path
        self.intermediate_save_frequency = intermediate_save_frequency

    def to_device(self, device: torch.types.Device):
        self.d = self.d.to(device=device)
        self.C = self.C.to(device=device)
        self.U = self.U.to(device=device)
        return self

    @torch.no_grad()
    def run(self, max_iter: int = 20, lambda_l: float = 0.1, lambda_s: float = 0.01, tol: float = 1e-5) \
            -> tuple[torch.Tensor, torch.Tensor]:
        """Run L+S reconstruction until convergence or for a specified maximum number of iterations.

        Args:
            max_iter (int, optional): Maximum number of iterations. Defaults to 20.
            lambda_l (float, optional): Soft threshold vor singular value thresholding. Defaults to 0.1.
            lambda_s (float, optional): Soft threshold in T space. Defaults to 0.01.
            tol (float, optional): Optimization stopping criterion. Defaults to 1e-5.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Low-rank and sparse components of reconstruction [t, x, y]
        """
        d = self.d.clone()  # [c, x*y, t]
        M = self._adjoint_op(d)  # [x*y, t]
        S = torch.zeros_like(M)
        L = M
        res = torch.inf
        pbar = tqdm(range(max_iter))
        for k in pbar:
            if res <= tol and k > 1:
                break
            Lk = self._svt(M-S, lambda_l)
            TSk = fftnc(M-L, dim=(1,))
            Sk = ifftnc(self._soft_thresholding(TSk, lambda_s), dim=(1,))  # type: torch.Tensor  # type: ignore
            Mk = Lk + Sk - self._adjoint_op(self._forward_op(Lk + Sk) - d)
            res = torch.linalg.norm(Lk + Sk - (L + S)) / torch.linalg.norm(L + S)
            pbar.set_postfix({'Residual': res.item()})
            M = Mk
            S = Sk
            L = Lk

            if self.intermediate_save_frequency > 0 and ((k+1) % self.intermediate_save_frequency == 0 or k == 0 or k == max_iter - 1):
                self._save_intermediate(L, S, k+1)

        L = L.reshape(*self.image_size, L.shape[1]).permute([2, 0, 1])  # [t, x, y]
        S = S.reshape(*self.image_size, S.shape[1]).permute([2, 0, 1])  # [t, x, y]
        return L + S

    def _soft_thresholding(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        """Soft thresholding operator

        Args:
            x (torch.Tensor): Array of arbitrary dimensions
            threshold (float): Soft threshold

        Returns:
            torch.Tensor: Thresholded array with same dimensions as `x`
        """
        x_out = x / (x.abs() + torch.finfo(torch.float64).eps) * torch.clamp(x.abs() - threshold, min=0)
        return x_out.to(x.dtype)

    def _svt(self, L: torch.Tensor, threshold: float) -> torch.Tensor:
        """Singular value thresholding operator

        Args:
            M (torch.Tensor): Low-rank image [x*y, t]
            threshold (float): Soft thresholding value

        Returns:
            torch.Tensor: Singular value thresholded image
        """
        U, S, Vh = torch.linalg.svd(L, full_matrices=False)
        S_thres = self._soft_thresholding(S, S[0] * threshold)
        return U @ torch.diag(S_thres + 0j) @ Vh

    def _forward_op(self, M: torch.Tensor) -> torch.Tensor:
        """Forward encoding operator

        Args:
            M (torch.Tensor): Image [x*y, t]

        Returns:
            torch.Tensor: k-Space representation [c, x*y, t]
        """
        CM = self.C * M
        CM = CM.reshape(CM.shape[0], *self.image_size, CM.shape[2])
        FCM = fftnc(CM, dim=(1, 2))
        FCM = FCM.reshape(FCM.shape[0], -1, FCM.shape[3])
        UFCM = self.U * FCM
        return UFCM

    def _adjoint_op(self, d: torch.Tensor) -> torch.Tensor:
        """Adjoing encoding operator

        Args:
            d (torch.Tensor): k-space measurements [c, x*y, t]

        Returns:
            torch.Tensor: Image representation [x*y, t]
        """
        d = d.reshape(d.shape[0], *self.image_size, d.shape[2])
        MC = ifftnc(d, dim=(1, 2))
        MC = MC.reshape(MC.shape[0], -1, MC.shape[3])
        M = (self.C.conj() * MC).sum(0)
        return M
    
    def _save_intermediate(self, L: torch.Tensor, S: torch.Tensor, iteration: int):
        """Save intermediate results of L+S reconstruction

        Args:
            L (torch.Tensor): Low-rank component 
            S (torch.Tensor): Sparse component
            iteration (int): Current iteration number
        """
        if self.intermediate_path is None: return

        l = L.reshape(*self.image_size, L.shape[1]).permute([2, 0, 1])  # [t, x, y]
        s = S.reshape(*self.image_size, S.shape[1]).permute([2, 0, 1])  # [t, x, y]
        reconstruction = l + s  # [t, x, y]
        torch.save(reconstruction, f'{self.intermediate_path}/reconstruction_{iteration}.pt')


