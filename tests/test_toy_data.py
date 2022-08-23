from .toy_dataset import Parameters as DatasetParameters
from .toy_dataset import create_synthetic_dataset
from dnmfx.component_description import create_component_description
from dnmfx.evaluate import evaluate
from dnmfx.fit import fit
from dnmfx.optimize import dnmf
from dnmfx.parameters import Parameters as DNMFXParameters
from dnmfx.utils import sigmoid
import jax.numpy as jnp
import math
import unittest


class TestCaseA(unittest.TestCase):

    def test_case_A(self):
        """
        Checking gradient computation on the simplest test case.
        """

        toy_data = self._generate_toy_data(
                           cell_centers= [(0.5, 0.5)],
                           image_size=1,
                           cell_size=1,
                           num_frames=1)

        H, W, B, log = fit(toy_data,
                           max_iteration=1,
                           batch_size=1,
                           log_every=2,
                           log_gradients=True)

        iter_log = log[0].iteration_logs[0]
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


    def test_case_B(self):
        """
        Checking if fitting converges on toy data.
        """
        cell_centers = [(32, 32)]
        toy_data = self._generate_toy_data(
                           cell_centers,
                           image_size=256,
                           cell_size=16,
                           num_frames=100)

        H, W, B, log = fit(toy_data,
                           max_iteration=100000,
                           min_loss=1e-1,
                           batch_size=10,
                           log_every=100,
                           log_gradients=False)
        print(f"log: {log}")
        last_iter_log = log[0].iteration_logs[-1]
        assert last_iter_log.iteration < 100000-1, \
            "Fitting should converges with the max num of iterations."

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
