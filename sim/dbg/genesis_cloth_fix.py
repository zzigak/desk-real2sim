"""Make Genesis's IPC cloth an isotropic membrane.

genesis/engine/couplers/ipc_coupler/coupler.py applies libuipc's StrainLimitingBaraffWitkinShell with a
single ElasticModuli2D, which libuipc uses for BOTH terms. Per libuipc's own docstring the stretch term
is E*t/(1-nu^2) (scales with thickness) while the shear term is E/(2(1+nu)) used literally (no
thickness). With t = 1 mm the shear stiffness therefore comes out 1/t = 1000x that of an isotropic
sheet with the same stretch stiffness: a cloth that stretches like rubber but will not shear, which
also stalls the solver (an overhanging strip with no bending stiffness at all crept 1.4 mm in 3 s
instead of hanging). We re-route the call to libuipc's two-moduli overload so that
    stretch = E_h / (1 - nu^2),   shear = E_h / (2 (1 + nu))     [N/m, E_h = E * thickness]
i.e. the isotropic plane-stress membrane Newton's VBD StVK gets from the same E_h and nu.
Import and call patch(E_h, nu) BEFORE scene.build(). Genesis itself is not modified.
"""
_PATCHED = {}


def patch(E_h, nu):
    from uipc.constitution import ElasticModuli2D, StrainLimitingBaraffWitkinShell

    if "orig" not in _PATCHED:
        _PATCHED["orig"] = StrainLimitingBaraffWitkinShell.apply_to
    orig = _PATCHED["orig"]

    def apply_to(self, sc, moduli=None, mass_density=200.0, thickness=0.001, strain_rate=100.0, **kw):
        # Both moduli are given as youngs_poisson pairs; libuipc turns them into the two stiffnesses above.
        stretch = ElasticModuli2D.youngs_poisson(E_h / thickness, nu)  # -> E_h/(1-nu^2)
        shear = ElasticModuli2D.youngs_poisson(E_h, nu)  # -> E_h/(2(1+nu)), literal
        _PATCHED["calls"] = _PATCHED.get("calls", 0) + 1
        return orig(self, sc, stretch, shear, mass_density, thickness, strain_rate)

    StrainLimitingBaraffWitkinShell.apply_to = apply_to
