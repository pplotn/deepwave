"""PyTorch Module and Function for acoustic wave propagator."""
import deepwave.base.propagator
import constant_density.PropagatorFunction
import variable_density.PropagatorFunction

class Propagator(deepwave.base.propagator.Propagator):
    """PyTorch Module for acoustic wave propagator.

    See deepwave.base.propagator.Propagator for description.
    """

    def __init__(self, model, dx, pml_width=None, survey_pad=None, vpmax=None):
        if list(model.keys()) == ['vp']:
            PropagatorFunction = constant_density.PropagatorFunction
        elif set(list(model.keys())) == set(['vp', 'rho']):
            PropagatorFunction = variable_density.PropagatorFunction
            if model['rho'].min() <= 0.0:
                raise RuntimeError('rho must be > 0, but min is {}'
                                   .format(model['rho'].min()))
        else:
            raise RuntimeError('Model must either only contain vp, '
                               'or vp and rho, but contains {}'
                               .format(list(model.keys())))

        if model['vp'].min() <= 0.0:
            raise RuntimeError('vp must be > 0, but min is {}'
                               .format(model['vp'].min()))

        super(Propagator, self).__init__(PropagatorFunction, model, dx,
                                         fd_width=2,  # also in Pml
                                         pml_width=pml_width,
                                         survey_pad=survey_pad)
        self.model.extra_info['vpmax'] = vpmax
