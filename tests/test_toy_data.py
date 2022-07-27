from dnmfx.utils import sigmoid
from dnmfx.optimize import dnmf
from dnmfx.component_description import create_component_description
from dnmfx.parameters import Parameters as DNMFXParameters
import jax.numpy as jnp
import math
import unittest
from .toy_dataset import create_synthetic_dataset
from .toy_dataset import Parameters as DatasetParameters


class TestCaseA(unittest.TestCase):

    def test_case_A(self):

        """
        Checking gradient computation on the simplest test case.
        """
        cell_centers = [(0.5, 0.5)]
        image_size = 1
        cell_size = 1
        num_frames = 1
        toy_data = self._generate_toy_data(
                           cell_centers,
                           image_size,
                           cell_size,
                           num_frames)
        H, W, B, log = self._fit(toy_data)

        iter_log = log.iteration_logs[0]
        w_logits = iter_log.W_logits[0][0]
        h_logits = iter_log.H_logits[0][0]
        b_logits = iter_log.B_logits[0][0]

        expected_grad_H_logits = \
                1/(1 + math.e**(-w_logits)) * \
                math.e**(-h_logits)/(1 + math.e**(-h_logits))**2

        expected_grad_W_logits = \
                1/(1 + math.e**(-h_logits)) * \
                math.e**(-w_logits)/(1 + math.e**(-w_logits))**2

        expected_grad_B_logits = math.e**(-b_logits)/(1 + math.e**(-b_logits))**2

        assert abs(iter_log.grad_H_logits - expected_grad_H_logits) < 1e-3, \
                "Partial gradient H wrt. loss matches expectation"
        assert abs(iter_log.grad_W_logits - expected_grad_W_logits) < 1e-3, \
                "Partial gradient W wrt. loss matches expectation"
        assert abs(iter_log.grad_B_logits - expected_grad_B_logits) < 1e-3, \
                "Partial gradient B wrt. loss matches expectation"


    def _generate_toy_data(self,
                           cell_centers,
                           image_size,
                           cell_size,
                           num_frames):

        dataset_parameters = DatasetParameters(image_size,
                                cell_size,
                                num_frames,
                                cell_centers)
        toy_data = create_synthetic_dataset(dataset_parameters)
        toy_data.sequence = toy_data.render()

        return toy_data


    def _fit(self, toy_data):

        component_descriptions = \
                        create_component_description(toy_data.bounding_boxes)
        dnmfx_parameters = DNMFXParameters(max_iteration=1, batch_size=1)
        H, W, B, log = dnmf(toy_data.sequence,
                            component_descriptions,
                            dnmfx_parameters,
                            log_every=1,
                            log_gradients=True,
                            random_seed=None)
        return H, W, B, log
