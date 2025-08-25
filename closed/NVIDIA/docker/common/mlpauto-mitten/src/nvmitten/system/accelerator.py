from __future__ import annotations
from dataclasses import dataclass, field
from typing import ClassVar, List, Type

from .component import Component


@dataclass
class Accelerator(Component):
    _registered: ClassVar[List] = []

    name: str

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        Accelerator._registered.append(cls)

    @classmethod
    def detect(cls) -> Dict[Type[Accelerator], List[Accelerator]]:
        detected = dict()
        for c in Accelerator._registered:
            detected[c] = c.detect()
        return detected


class NUMASupported:
    """Mixin class to help with attaching accelerators to host NUMA nodes.
    """

    @property
    def numa_host_id(self) -> int:
        """The host NUMA node ID affiliated with this device.

        Returns:
            int: None if NUMA is supported but not enabled. Otherwise returns the host NUMA node ID as an int.
        """
        return None
