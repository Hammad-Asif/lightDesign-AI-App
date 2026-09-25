"""Roofline: roof structure lines from a street-level photo.

Masks are plane support regions, not the source of lines. Lines come from (a) plane-to-plane
adjacency after a hard partition and (b) robust line fits over trusted boundary arcs.
"""
