# -*- coding: utf-8 -*-

from __future__ import absolute_import

from colour.utilities import CaseInsensitiveMapping, filter_kwargs

from .common import (
    Jab_to_spherical, spherical_to_Jab, close_gamut_boundary_descriptor,
    fill_gamut_boundary_descriptor, sample_gamut_boundary_descriptor,
    tessellate_gamut_boundary_descriptor)
from .morovic2000 import gamut_boundary_descriptor_Morovic2000

__all__ = [
    'Jab_to_spherical', 'spherical_to_Jab', 'close_gamut_boundary_descriptor',
    'fill_gamut_boundary_descriptor', 'sample_gamut_boundary_descriptor',
    'tessellate_gamut_boundary_descriptor'
]
__all__ += ['gamut_boundary_descriptor_Morovic2000']

GAMUT_BOUNDARY_DESCRIPTOR_METHODS = CaseInsensitiveMapping({
    'Morovic 2000': gamut_boundary_descriptor_Morovic2000
})
GAMUT_BOUNDARY_DESCRIPTOR_METHODS.__doc__ = """
Supported *Gamut Boundary Descriptor (GDB)* computation methods.

References
----------
:cite:``

GAMUT_BOUNDARY_DESCRIPTOR_METHODS : CaseInsensitiveMapping
    **{'Morovic 2000'}**
"""


def gamut_boundary_descriptor(Jab, method='Morovic 2000', **kwargs):
    function = GAMUT_BOUNDARY_DESCRIPTOR_METHODS[method]

    return function(Jab, **filter_kwargs(function, **kwargs))


__all__ += ['GAMUT_BOUNDARY_DESCRIPTOR_METHODS', 'gamut_boundary_descriptor']
