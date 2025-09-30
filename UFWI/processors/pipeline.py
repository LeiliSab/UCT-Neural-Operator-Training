"""
UFWI.pipeline
====================

Defines :class:`Pipeline`, a thin convenience wrapper that executes a list of
processors in sequence.  The same :class:`~UFWI.data.AcquisitionData`
instance is threaded through all stages and returned at the end, so memory
usage never grows with the pipeline depth.
"""
import time as time
from typing import Iterable, List, Optional, Dict, Any

from ..data import AcquisitionData
from .base import BaseProcessor


class Pipeline:
    """
    Sequential data-flow executor.

    Parameters
    ----------
    stages
        Iterable of processors (executed in order).
    verbose
        If True prints progress to ``stdout``.
    logger
        Optional :pyclass:`logging.Logger`‐like object.  If supplied, all
        messages are sent to ``logger.info`` **instead** of ``stdout``.

    Example
    -------
     preprocess = Pipeline(
                stages=[
                TimeWindow(),
                DTFT(freqs),
                PhaseScreenCorrection(img_geom),
                DownSample(step=1),
                AcceptanceMask(delta=63),
                MagnitudeOutlierFilter(keep_fraction=0.99),
            ],
            verbose=True
        )
     acq_data = preprocess(acq_data)
    """

    def __init__(self,
                 stages: Iterable[BaseProcessor],
                 *,
                 verbose: bool = False,
                 logger: Any | None = None) -> None:
        self.stages: List[BaseProcessor] = list(stages)
        self._verbose = verbose
        self._logger = logger

    def __call__(self,
                 data: AcquisitionData,
                 ctx: Optional[Dict[str, Any]] = None) -> AcquisitionData:
        """
        Execute the pipeline on *data*.

        Parameters
        ----------
        data
            The one and only :class:`~UFWI.data.AcquisitionData`
            instance flowing through the pipeline.
        ctx
            Optional dictionary merged into ``data.ctx`` *before* the first
            processor is run.  You rarely need this – processors now carry
            their own dependencies – but it remains available for ad-hoc
            metadata injection.

        Returns
        -------
        AcquisitionData
            The same object after in-place modification.
        """

        if ctx:
            data.ctx.update(ctx)

        for i, stage in enumerate(self.stages, start=1):
            name = stage.__class__.__name__
            self._log(f"[{i:02}/{len(self.stages)}] {name} …")
            tic = time.perf_counter()

            stage(data)  # in-place

            toc = time.perf_counter() - tic
            self._log(f"    done in {toc:6.3f} s")

        return data

    def _log(self, msg: str) -> None:

        if self._logger is not None:
            self._logger.info(msg)
        elif self._verbose:
            print(msg)
