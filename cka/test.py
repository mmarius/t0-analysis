import sys

sys.path.insert(1, "/t0-analysis/CKA-similarity")

import numpy as np

from original_cka import gram_linear, gram_rbf, cca, cka, feature_space_linear_cka
from CKA import CKA


def check_invariance_properties(X, Y):
    transform = np.random.randn(10, 10)
    _, orthogonal_transform = np.linalg.eigh(transform.T.dot(transform))

    # CKA is invariant only to orthogonal transformations.
    np.testing.assert_almost_equal(
        feature_space_linear_cka(X, Y),
        feature_space_linear_cka(X.dot(orthogonal_transform), Y),
    )
    np.testing.assert_(
        not np.isclose(
            feature_space_linear_cka(X, Y),
            feature_space_linear_cka(X.dot(transform), Y),
        )
    )

    # CCA is invariant to any invertible linear transform.
    np.testing.assert_almost_equal(cca(X, Y), cca(X.dot(orthogonal_transform), Y))
    np.testing.assert_almost_equal(cca(X, Y), cca(X.dot(transform), Y))

    # Both CCA and CKA are invariant to isotropic scaling.
    np.testing.assert_almost_equal(cca(X, Y), cca(X * 1.337, Y))
    np.testing.assert_almost_equal(
        feature_space_linear_cka(X, Y), feature_space_linear_cka(X * 1.337, Y)
    )


def main():
    np.random.seed(1337)
    X = np.random.randn(100, 10)
    Y = np.random.randn(100, 10) + X

    cka_from_examples = cka(gram_linear(X), gram_linear(Y))
    cka_from_features = feature_space_linear_cka(X, Y)

    print("Linear CKA from Examples: {:.5f}".format(cka_from_examples))
    print("Linear CKA from Features: {:.5f}".format(cka_from_features))
    np.testing.assert_almost_equal(cka_from_examples, cka_from_features)

    np_cka = CKA()
    cka_score = np_cka.linear_CKA(X, Y)
    print("Linear CKA (re-implementation): {:.5f}".format(cka_score))
    np.testing.assert_almost_equal(cka_from_examples, cka_score)

    # compute CKA with nonlinear kernels
    rbf_cka = cka(gram_rbf(X, 0.5), gram_rbf(Y, 0.5))
    print("RBF CKA: {:.5f}".format(rbf_cka))

    # TODO(mm): What's the right sigma here to get the same result as above?
    rbf_cka_score = np_cka.kernel_CKA(X, Y, sigma=None)
    print("RBF CKA (re-implementation): {:.5f}".format(rbf_cka_score))

    # compute a "debiased" form of CKA
    cka_from_examples_debiased = cka(gram_linear(X), gram_linear(Y), debiased=True)
    cka_from_features_debiased = feature_space_linear_cka(X, Y, debiased=True)

    print(
        "Linear CKA from Examples (Debiased): {:.5f}".format(cka_from_examples_debiased)
    )
    print(
        "Linear CKA from Features (Debiased): {:.5f}".format(cka_from_features_debiased)
    )

    np.testing.assert_almost_equal(
        cka_from_examples_debiased, cka_from_features_debiased
    )

    # check invariance properties
    check_invariance_properties(X, Y)


if __name__ == "__main__":
    main()
