"""Per-material version of ~/projects/franka-deformables/genesis_cloth_fix.py.

Genesis hands libuipc's StrainLimitingBaraffWitkinShell ONE ElasticModuli2D for stretch and shear, which makes
shear 1/thickness too stiff. This reroutes each call to the two-moduli overload:
    stretch = E_h / (1 - nu^2),  shear = E_h / (2 (1 + nu)),  E_h = E * thickness   [N/m]
Materials are told apart by thickness (cloth 1 mm, paper 0.1 mm, pad 3 mm). Call patch() before scene.build()."""
_P = {}

def patch(table):
    """table: {thickness: (E_h, nu)}."""
    from uipc.constitution import ElasticModuli2D, StrainLimitingBaraffWitkinShell
    _P.setdefault("orig", StrainLimitingBaraffWitkinShell.apply_to)
    orig = _P["orig"]

    def apply_to(self, sc, moduli=None, mass_density=200.0, thickness=0.001, strain_rate=100.0, **kw):
        t = min(table, key=lambda k: abs(k - thickness))
        E_h, nu = table[t]
        _P.setdefault("calls", []).append(t)
        return orig(self, sc, ElasticModuli2D.youngs_poisson(E_h / thickness, nu), ElasticModuli2D.youngs_poisson(E_h, nu),
                    mass_density, thickness, strain_rate)

    StrainLimitingBaraffWitkinShell.apply_to = apply_to
