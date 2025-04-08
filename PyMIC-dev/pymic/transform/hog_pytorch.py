from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from functools import lru_cache
import torch
import math
import torch.nn.functional as F
from typing import Literal


class HOGVisualization:
    @lru_cache(maxsize=1)
    def _generateAngleLines(self, phaseMin=10, phaseMax=170):
        """
        Generates angle lines for visualization.

        Args:
            phaseMin (int, optional): The minimum phase value. Defaults to 10.
            phaseMax (int, optional): The maximum phase value. Defaults to 170.
            cellSize (int, optional): The size of each cell in pixels. 
            Defaults to 16.

        Returns:
            torch.Tensor: The generated angle lines.
        """
        angle = (torch.linspace(phaseMin, phaseMax, 9)) / 180 * math.pi
        zeroPhaseLine = torch.zeros(self.cellSize, self.cellSize)
        centerCellSize = self.cellSize // 2
        zeroPhaseLine[centerCellSize] = 1
        coords = torch.dstack(
            torch.meshgrid(
                *2 * [torch.linspace(-1, 1, self.cellSize).to(torch.float32)], 
                indexing="xy"
            )
        )
        rotMat = torch.stack(
            (
                torch.cos(angle), -torch.sin(angle),
                torch.sin(angle), torch.cos(angle)
            ), dim=-1
        ).view(-1, 2, 2)
        return (
            torch.nn.functional.grid_sample(
                zeroPhaseLine.expand(angle.size(0), 1, -1, -1),
                torch.einsum("ijk,bkl->bijl", coords, rotMat),
                align_corners=False,
                mode="bilinear",
            )
            .moveaxis(0, -1)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
        )

class HOGParametrization:
    SOBEL_KERNEL = torch.tensor(
        [
            [-1.0 - 1.0j, 0.0 - 2.0j, 1.0 - 1.0j],
            [-2.0 + 0.0j, 0.0 + 0.0j, 2.0 + 0.0j],
            [-1.0 + 1.0j, 0.0 + 2.0j, 1.0 + 1.0j],
        ],
        dtype=torch.complex64,
    )

    def _chooseChannelsReduceFunc(self, channelWise: bool):
        def channelsMax(norm, grads):
            return (
                torch.gather(
                    norm,
                    1,
                    normAMaxChan := norm.argmax(dim=1, keepdim=True)
                ),
                torch.gather(grads, 1, normAMaxChan),
            )

        def channelsIdentity(norm, grads):
            return norm, grads

        if channelWise:
            return channelsIdentity
        else:
            return channelsMax

    def _chooseAccumulateFunc(self, accumulate: Literal["simple", "bilinear"]):
        def accumulateBilinear(zeros, leftEdgeIndices, norm, phase, binsEdges):
            leftEdgePrct = self.phaseGain * (
                phase - binsEdges[leftEdgeIndices]
            )  # Compute left edge distance percentage
            return zeros.scatter_add(
                -1,
                leftEdgeIndices % self.nPhaseBins,
                (1 - leftEdgePrct) * norm,
            ).scatter_add(  # Add weighted norm to left edge
                -1, (leftEdgeIndices + 1) % self.nPhaseBins,
                leftEdgePrct * norm
            )  # Add weighted norm to right edge

        return accumulateBilinear


    def _chooseGradFunc(self, kernel: Literal["sobel", "finite"]):
        def gradSobel(im):
            kernel = HOGParametrization.SOBEL_KERNEL.expand(
                im.size(1), 1, -1, -1
            )
            return F.conv2d(
                torch.nn.functional.pad(
                    im.to(kernel.dtype), 4 * [kernel.shape[0] // 2],
                    mode="reflect"
                ),
                kernel,
                groups=im.size(1),
            )
        return gradSobel


    def _chooseNormalizationFunc(self, norm: Literal["L1", "L2"]):
        def normL2Hys(x):
            return (
                trhold := (
                    x / (x.norm(p=2, dim=(-1, -2, -3), keepdim=True) + 1e-10)
                ).clamp(max=0.2)
            ) / (trhold.norm(p=2, dim=(-1, -2, -3), keepdim=True) + 1e-10)

        return normL2Hys

class HOG(torch.nn.Module, HOGVisualization, HOGParametrization):
    """
    HOG (Histogram of Oriented Gradients) module.

    Args:
        cellSize (int): Size of the cell in pixels. Default is 16.
        blockSize (int): Size of the block in cells. Default is 2.
        nPhaseBins (int): Number of phase bins. Default is 9.
        kernel (Literal["sobel", "finite"]): Type of gradient kernel.
            Default is "finite".
        normalization (Literal["L2", "L1", "L2Hys"]): Type of normalization.
            Default is "L2".
        accumulate (Literal["simple", "bilinear"]): Type of accumulation.
            Default is "bilinear".
        channelWise (bool): Whether to perform channel-wise reduction.
            Default is False.
    """

    def __init__(
        self,
        cellSize: int = 16,
        blockSize: int = 2,
        nPhaseBins: int = 9,
        kernel: Literal["sobel", "finite"] = "finite",
        normalization: Literal["L2", "L1", "L2Hys"] = "L2",
        accumulate: Literal["simple", "bilinear"] = "bilinear",
        channelWise: bool = False,
    ):
        super(HOG, self).__init__()
        # Initialize parameters
        if not isinstance(cellSize, int) or cellSize < 1:
            raise ValueError("cellSize must be a positive integer")
        self.cellSize = cellSize
        if not isinstance(blockSize, int) or blockSize < 1:
            raise ValueError("blockSize must be a positive integer")
        self.blockSize = blockSize
        if not isinstance(nPhaseBins, int) or nPhaseBins < 1:
            raise ValueError("nPhaseBins must be an integer >1")
        self.nPhaseBins = nPhaseBins
        self.phaseGain = self.nPhaseBins / math.pi
        if not isinstance(accumulate, str) or accumulate not in [
            "simple", "bilinear"
        ]:
            raise ValueError(
                "accumulate must be either 'simple' or 'bilinear'"
            )
        self.accumulate = self._chooseAccumulateFunc(accumulate)
        if not isinstance(channelWise, bool):
            raise ValueError("channelWise must be a boolean")
        self.channelWise = self._chooseChannelsReduceFunc(channelWise)
        self.grad = self._chooseGradFunc(kernel)
        self.normalization = self._chooseNormalizationFunc(normalization)

    def forward(self, im: torch.Tensor):
        """
        Forward pass of the HOG module.

        Args:
            im (torch.Tensor): Input image tensor (B,C,H,W).

        Returns:
            torch.Tensor: Normalized HOG features
                (H//cellSize, W//cellSize, blockSize, blockSize, nPhaseBins)
        """
        # Compute cells histograms, unfold to get blocks and then normalize 
        # per block
        return self.blockNormalize(self.hog(im))

    def hog(self, im: torch.Tensor):
        """
        Compute HOG features for the input image.

        Args:
            im (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: HOG features.
        """
        # Compute gradients and then unfold to get cells
        grads = (
            self.grad(im)
            .unfold(-2, *2 * [self.cellSize])
            .unfold(-2, *2 * [self.cellSize])
            .flatten(-2, -1)
        )
        # Compute norm and reduce channel dimension if asked
        norm, grads = self.channelWise(grads.abs(), grads)
        return self.cellsHist(norm, grads.angle() % math.pi)

    def cellsHist(self, norm, phase):
        """
        Compute histograms of cells.

        Args:
            norm (torch.Tensor): Norm of gradients.
            phase (torch.Tensor): Phase of gradients.

        Returns:
            torch.Tensor: Histograms of cells.
        """
        binsEdges = torch.linspace(0, math.pi, self.nPhaseBins + 1)
        leftEdgeIndices = (self.phaseGain * phase).floor().to(torch.int64)
        return (
            self.accumulate(
                torch.zeros((*phase.shape[:-1], self.nPhaseBins)),
                leftEdgeIndices,
                norm,
                phase,
                binsEdges,
            )
            / self.cellSize**2
        )

    def blockNormalize(self, hog: torch.Tensor):
        """
        Normalize HOG features per block.

        Args:
            hog (torch.Tensor): HOG features.

        Returns:
            torch.Tensor: Normalized HOG features.
        """
        return self.normalization(
            hog.unfold(
                -3, self.blockSize, halfBlockSize := self.blockSize // 2
            )
            .unfold(-3, self.blockSize, halfBlockSize)
            .moveaxis(-3, -1)
        )

