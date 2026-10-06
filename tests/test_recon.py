import numpy as np
import pytest
import torch

from slam_t2_data.recon import cg_sense, fft2c, ifft2c, sense_adjoint, sense_forward


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def random_complex(shape, seed=1):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, dtype=torch.complex128, generator=generator)


@pytest.mark.parametrize("shape", [(4, 6), (5, 7)])
def test_fourier_matches_explicit_centered_dft(shape):
    h, w = shape
    x = random_complex(shape)
    # Coordinates measured relative to floor(N/2), including odd lengths.
    a = np.arange(h) - h // 2
    b = np.arange(w) - w // 2
    fx = np.exp(-2j * np.pi * np.outer(a, a) / h) / np.sqrt(h)
    fy = np.exp(-2j * np.pi * np.outer(b, b) / w) / np.sqrt(w)
    expected = fx @ x.numpy() @ fy.T
    np.testing.assert_allclose(fft2c(x).numpy(), expected, atol=2e-14, rtol=2e-14)
    torch.testing.assert_close(ifft2c(fft2c(x)), x, atol=2e-14, rtol=2e-14)


@pytest.mark.parametrize("shape", [(4, 6), (5, 7)])
def test_complex_adjoint(shape):
    x = random_complex((2, *shape))
    maps = random_complex((2, 3, *shape), 2)
    y = random_complex(maps.shape, 3)
    mask = torch.arange(shape[1]) % 2 == 0
    left = torch.vdot(sense_forward(x, maps, mask).flatten(), y.flatten())
    right = torch.vdot(x.flatten(), sense_adjoint(y, maps, mask).flatten())
    torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("lam", [0.0, 0.2])
@pytest.mark.parametrize("warm_start", [False, True])
def test_cg_matches_independent_dense_solve_and_preserves_inputs(lam, warm_start):
    h, w, c = 3, 4, 2
    maps = random_complex((1, c, h, w), 4)
    y = random_complex(maps.shape, 5)
    mask = torch.tensor([True, False, True, True])
    original = y.clone()
    # Assemble a dense forward matrix independently with NumPy FFTs.
    columns = []
    for i in range(h * w):
        basis = np.eye(h * w, dtype=np.complex128)[i].reshape(h, w)
        coil_images = maps[0].numpy() * basis
        encoded = np.fft.fftshift(
            np.fft.fft2(np.fft.ifftshift(coil_images, axes=(-2, -1)), norm="ortho"),
            axes=(-2, -1),
        )
        columns.append((encoded * mask.numpy()).ravel())
    a = np.stack(columns, axis=1)
    expected = np.linalg.solve(
        a.conj().T @ a + lam * np.eye(h * w), a.conj().T @ y.numpy().ravel()
    ).reshape(h, w)
    x0 = random_complex((1, h, w), 17) if warm_start else None
    initial = None if x0 is None else x0.clone()
    image, residual, _ = cg_sense(y, maps, mask, lam=lam, iters=100, tol=1e-12, x0=x0)
    np.testing.assert_allclose(image[0].numpy(), expected, atol=1e-10, rtol=1e-10)
    assert residual.item() < 1e-11
    torch.testing.assert_close(y, original, atol=0, rtol=0)
    if x0 is not None:
        torch.testing.assert_close(x0, initial, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("lam", [0.0, 0.2])
def test_zero_rhs_with_nonzero_initial_image(dtype, lam):
    maps = random_complex((2, 2, 4, 6), 83).to(dtype)
    y = random_complex(maps.shape, 84).to(dtype)
    y[0] = 0
    initial = random_complex((2, 4, 6), 85).to(dtype)
    mask = torch.ones(6, dtype=torch.bool)
    image, residual, steps = cg_sense(y, maps, mask, lam=lam, iters=100, x0=initial)
    torch.testing.assert_close(image[0], torch.zeros_like(image[0]), atol=0, rtol=0)
    assert residual[0] == 0 and steps[0] == 0
    assert residual[1] < 1e-5 and steps[1] > 0


def test_batched_slices_match_independent_solves():
    maps = random_complex((3, 2, 4, 6), 21)
    maps[0] = 1
    y = random_complex(maps.shape, 22)
    mask = torch.tensor([True, False, True, False, True, True])
    image, residual, steps = cg_sense(y, maps, mask, lam=0.2, iters=100, tol=1e-12)
    assert steps[0] == 1 and (steps[1:] > 1).all()
    for i in range(3):
        single, single_residual, single_steps = cg_sense(
            y[i : i + 1], maps[i : i + 1], mask, lam=0.2, iters=100, tol=1e-12
        )
        torch.testing.assert_close(image[i], single[0], atol=1e-11, rtol=1e-11)
        torch.testing.assert_close(residual[i], single_residual[0], atol=1e-12, rtol=0)
        assert steps[i] == single_steps[0]


def test_unregularized_reconstruction_with_zero_map_support():
    maps = random_complex((1, 2, 4, 6), 31)
    maps[..., 0, :] = 0
    target = random_complex((1, 4, 6), 32)
    mask = torch.ones(6, dtype=torch.bool)
    y = sense_forward(target, maps, mask)
    expected = target.clone()
    expected[:, 0, :] = 0
    image, residual, _ = cg_sense(y, maps, mask, iters=100, tol=1e-12)
    torch.testing.assert_close(image, expected, atol=1e-10, rtol=1e-10)
    assert residual.item() < 1e-11


def test_mixed_zero_rhs_and_initial_solution():
    maps = torch.ones((2, 1, 4, 6), dtype=torch.complex128)
    x = random_complex((2, 4, 6), 8)
    x[0] = 0
    mask = torch.ones(6, dtype=torch.bool)
    y = sense_forward(x, maps, mask)
    image, residual, steps = cg_sense(y, maps, mask, tol=1e-12)
    torch.testing.assert_close(image, x, atol=1e-12, rtol=1e-12)
    assert steps.tolist() == [0, 1]
    assert (residual < 1e-12).all()
    again, _, steps = cg_sense(y, maps, mask, tol=1e-12, x0=x)
    torch.testing.assert_close(again, x)
    assert steps.tolist() == [0, 0]


def test_iteration_budget_and_invalid_inputs():
    maps = random_complex((1, 2, 4, 6), 9)
    y = random_complex(maps.shape, 10)
    mask = torch.tensor([True, False, True, False, True, True])
    _, residual, steps = cg_sense(y, maps, mask, iters=1, tol=1e-12)
    assert residual.item() > 1e-12 and steps.item() == 1
    with pytest.raises(ValueError, match="mask"):
        cg_sense(y, maps, mask.float())
    with pytest.raises(ValueError, match="lambda"):
        cg_sense(y, maps, mask, lam=-1)
    with pytest.raises(ValueError, match="finite"):
        cg_sense(y * float("nan"), maps, mask)
