# SPDX-License-Identifier: AGPL-3.0-only
def symlink_or_skip(case, link, target, **kwargs):
    """Keep real link tests enabled whenever the host grants link creation."""
    try:
        link.symlink_to(target, **kwargs)
    except OSError as error:
        if getattr(error, 'winerror', None) == 1314:
            case.skipTest('Windows account cannot create symbolic links; no permissions were changed')
        raise
