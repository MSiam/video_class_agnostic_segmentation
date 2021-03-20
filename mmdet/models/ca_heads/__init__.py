
from .compose_ca import ComposedClassAgnosticHead
from .appearance_ca_mahalanobis import MahalanobisAppearanceBasedClassAgnosticHead
from .appearance_ca_map import MAPClassAgnosticHead

__all__ = ['ComposedClassAgnosticHead',
           'MahalanobisAppearanceBasedClassAgnosticHead',
           'MAPClassAgnosticHead',
          ]
