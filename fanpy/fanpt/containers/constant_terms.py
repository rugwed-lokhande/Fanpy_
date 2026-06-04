r"""Class that generates and contains the constant terms of the FANPT system of equations."""

from math import factorial
import numpy as np

from fanpy.fanpt.containers.base import FANPTContainer


class FANPTConstantTerms:
    r"""Generates and contains the constant terms of the FANPT system of equations.

    If the order is 1:
    -dG_n/dl

    If the energy is not an active parameter:
    -N * sum_k {d^2G_n/dldp_k * d^(N-1)p_k/dl^(N-1)}

    If the energy is an active parameter:
    linear_term = t1 + t2
    t1 = -N * sum_k {d^2G_n/dldp_k * d^(N-1)p_k/dl^(N-1)}
    t2 = - sum_k{d^2G_n/dEdp_k * sum_m{C(N,m) * d^mE/dl^m * d^(N-m)p_k/dl^(N-m)}}

    If `quasi_approximation_order == 3`, additional second- and third-derivative
    contributions are included for orders 2 and 3 (when energy is active).

    Notation
    --------
    N: order
    sum_k: sum over all active wfn parameters.
    sum_m: sum from 1 to N-1.
    C(N,m): binomial coefficient.

    Attributes
    ----------
    fanpt_container : FANPTContainer
        Object containing the FANPT matrices and vectors.
    order : int
        Order of the current FANPT iteration.
    previous_responses : np.ndarray
        Previous responses of the FANPT calculations up to order = order - 1.
        Shape: (order-1, nactive)
    quasi_approximation_order : int
        Either 2 or 3. Controls inclusion of higher-derivative terms.

    Methods
    -------
    __init__(self, fanpt_container, order=1, previous_responses=None, quasi_approximation_order=2)
        Initialize the constant terms.
    assign_fanpt_container(self, fanpt_container):
        Assign the FANPT Container
    assign_order(self, order):
        Assign the order
    assign_previous_responses(self, previous_responses):
        Assign the previous responses
    assign_quasi_approximation_order(self, qao):
        Assign the quasi approximation order
    gen_constant_terms(self)
    """

    def __init__(self, fanpt_container, order=1, previous_responses=None, quasi_approximation_order=2):
        r"""Initialize the constant terms.

        Parameters
        ----------
        fanpt_container : FANPTContainer
            Object containing the FANPT matrices and vectors.
        order : int, optional
            Order of the current FANPT iteration (default: 1).
        previous_responses : np.ndarray or None, optional
            Previous responses of the FANPT calculations up to order = order - 1.
            If provided, expected shape is (order - 1, nactive). Default: None.
        quasi_approximation_order : int, optional
            2 or 3. If 3, include extra second-/third-derivative contributions
            (when energy is active) as described below. Default: 2.
        """
        self.assign_fanpt_container(fanpt_container=fanpt_container)
        self.assign_order(order=order)
        self.assign_quasi_approximation_order(quasi_approximation_order)
        self.assign_previous_responses(previous_responses=previous_responses)
        self.gen_constant_terms()

    def assign_fanpt_container(self, fanpt_container):
        r"""Assign the FANPT container.

        Parameters
        ----------
        fanpt_container : FANPTContainer
            Container with the matrices/vectors required to perform the FANPT calculation.

        Raises
        ------
        TypeError
            If fanpt_container is not a child of FANPTContainer.
        """
        if not isinstance(fanpt_container, FANPTContainer):
            raise TypeError("fanpt_container must be a child of FANPTContainer")
        self.fanpt_container = fanpt_container

    def assign_order(self, order):
        r"""Assign the order.

        Parameters
        ----------
        order : int
            Order of the current FANPT iteration.

        Raises
        ------
        TypeError
            If order is not an int.
        ValueError
            If order is negative.
        """
        if not isinstance(order, int):
            raise TypeError("order must be an integer.")
        if order < 0:
            raise ValueError("order must be non-negative")
        self.order = order

    def assign_quasi_approximation_order(self, qao):
        r"""Assign and validate the quasi approximation order.

        Parameters
        ----------
        qao : int
            Allowed values: 2 or 3.

        Raises
        ------
        ValueError
            If qao is not 2 or 3.
        """
        if qao is None:
            qao = 2

        if qao not in (2, 3):
            raise ValueError("quasi_approximation_order must be 2 or 3.")
        self.quasi_approximation_order = qao

    def assign_previous_responses(self, previous_responses):
        r"""Assign the previous responses.

        Parameters
        ----------
        previous_responses : np.ndarray
            Previous responses of the FANPT calculations up to order = order -1.

        Raises
        ------
        TypeError
            If previous_responses is not a numpy array.
            If the elements of previous_responses are not numpy arrays.
        ValueError
            If previous_responses is None and order is not 1.
            If the shape of previous_responses is not equal to (order - 1, nactive).
        """
        if self.order == 1:
            self.previous_responses = previous_responses
        else:
            if not isinstance(previous_responses, np.ndarray):
                raise TypeError("previous_responses must be a numpy array.")
            if not all([isinstance(response, np.ndarray) for response in previous_responses]):
                raise TypeError("The elements of previous_responses must be numpy arrays.")
            if previous_responses.shape != (self.order - 1, self.fanpt_container.nactive):
                raise ValueError(
                    "The shape of previous_responses must be ({}, {}).".format(
                        self.order - 1, self.fanpt_container.nactive
                    )
                )
            self.previous_responses = previous_responses



    def gen_constant_terms(self):
        r"""Generate the constant terms.
    
        Returns
        -------
        constant_terms : np.ndarray
            Constant terms of the FANPT linear system of equations.
            Shape: (nequation,)
        """
    
        def print_vector_diagnostics(name, vec, top_n=10):
            """Print basic diagnostics for a vector."""
            abs_vec = np.abs(vec)
            argmax = np.argmax(abs_vec)
    
            print(f"\n{name} diagnostics")
            print(f"  ||{name}||     =", np.linalg.norm(vec))
            print(f"  max|{name}|    =", np.max(abs_vec))
            print(f"  argmax|{name}| =", argmax)
            print(f"  {name}[argmax] =", vec[argmax])
    
            print(f"\nLargest {name} components")
            largest_indices = np.argsort(abs_vec)[-top_n:][::-1]
            for idx in largest_indices:
                print(
                    f"  idx={idx:6d}",
                    f"{name}={vec[idx]: .12e}",
                )
    
        def print_header(title):
            """Print a diagnostic section header."""
            print("\n" + "=" * 90)
            print(title)
            print("lambda =", self.fanpt_container.l)
    
        def print_footer():
            """Print a diagnostic section footer."""
            print("=" * 90 + "\n")
    
        # -------------------------------------------------------------------------
        # Order 1: no previous responses are needed.
        # -------------------------------------------------------------------------
        if self.order == 1:
            constant_terms = -self.fanpt_container.d_g_lambda
    
            print_header("ORDER 1 CONSTANT TERM / RHS DIAGNOSTICS")
            print_vector_diagnostics("-G_lambda", constant_terms)
            print_footer()
    
            self.constant_terms = constant_terms
            return constant_terms
    
        # -------------------------------------------------------------------------
        # Higher orders, energy is not active.
        # -------------------------------------------------------------------------
        if not self.fanpt_container.active_energy:
            constant_terms = -self.order * np.dot(
                self.fanpt_container.d2_g_lambda_wfnparams,
                self.previous_responses[-1],
            )
    
            self.constant_terms = constant_terms
            return constant_terms
    
        # -------------------------------------------------------------------------
        # Higher orders, active energy.
        #
        # Active-energy response convention:
        # response[:-1] = wavefunction-parameter response
        # response[-1]  = energy response
        # -------------------------------------------------------------------------
    
        # -------------------------------------------------------------------------
        # Build r_vector:
        #
        # r_k = sum_o C(N,o) E^(o) p_k^(N-o)
        #
        # where N = self.order and o = 1,...,N-1.
        # -------------------------------------------------------------------------
        r_vector = np.zeros(self.fanpt_container.nactive - 1)
    
        for o in range(1, self.order):
            comb = factorial(self.order) / (
                factorial(o) * factorial(self.order - o)
            )
    
            energy_response_o = self.previous_responses[o - 1][-1]
            wfn_response_n_minus_o = self.previous_responses[self.order - o - 1][:-1]
    
            r_vector += comb * energy_response_o * wfn_response_n_minus_o
    
        # -------------------------------------------------------------------------
        # Base constant terms.
        # These are the QAO=2-like terms and are always included for active-energy
        # FANPT, regardless of whether QAO=2 or QAO=3 is requested.
        # -------------------------------------------------------------------------
        wfn_response_prev = self.previous_responses[-1][:-1]
    
        lambda_wfn_term = -self.order * np.dot(
            self.fanpt_container.d2_g_lambda_wfnparams,
            wfn_response_prev,
        )
    
        energy_wfn_term = -np.dot(
            self.fanpt_container.d2_g_e_wfnparams,
            r_vector,
        )
    
        constant_terms = lambda_wfn_term + energy_wfn_term
        base_constant_terms = constant_terms.copy()
    
        # First-order response diagnostics are useful for both QAO=2 and QAO=3.
        p1 = self.previous_responses[0][:-1]
        e1 = self.previous_responses[0][-1]
    
        print_header(f"ORDER {self.order} BASE / QAO{self.quasi_approximation_order} DIAGNOSTICS")
    
        print("\nBase constant-term diagnostics")
        print("  ||base_constant_terms||  =", np.linalg.norm(base_constant_terms))
        print("  max|base_constant_terms| =", np.max(np.abs(base_constant_terms)))
        print("  argmax|base_constant_terms| =", np.argmax(np.abs(base_constant_terms)))
    
        print_vector_diagnostics("p1", p1)
        print("\nEnergy response diagnostics")
        print("  E1 =", e1)
    
        print_footer()
    
        # -------------------------------------------------------------------------
        # Optional QAO=3 contributions.
        # -------------------------------------------------------------------------
        if self.quasi_approximation_order == 3:
    
            # ---------------------------------------------------------------------
            # Order 2 QAO=3 term:
            #
            # - sum_kl d^2G / dp_k dp_l * p1_k * p1_l
            # ---------------------------------------------------------------------
            if self.order == 2:
                Gpp = self.fanpt_container.d2_g_wfnparams2
    
                qao3_term = -np.einsum(
                    "mkl,k,l->m",
                    Gpp,
                    p1,
                    p1,
                )
    
                base_norm = np.linalg.norm(base_constant_terms)
                qao3_norm = np.linalg.norm(qao3_term)
    
                print_header("ORDER 2 QAO3 DIAGNOSTICS")
    
                print("\nConstant-term norms")
                print("  ||base||      =", base_norm)
                print("  ||qao3_term|| =", qao3_norm)
                print(
                    "  ratio         =",
                    qao3_norm / base_norm if base_norm != 0 else np.inf,
                )
    
                print_vector_diagnostics("qao3_term", qao3_term)
                print_vector_diagnostics("p1", p1)
    
                print("\nG_pp tensor diagnostics")
                print("  Gpp shape     =", Gpp.shape)
                print("  ||Gpp||       =", np.linalg.norm(Gpp))
                print("  max|Gpp|      =", np.max(np.abs(Gpp)))
                print(
                    "  argmax|Gpp|   =",
                    np.unravel_index(np.argmax(np.abs(Gpp)), Gpp.shape),
                )
    
                Gpp_norm = np.linalg.norm(Gpp)
                sym_error = np.linalg.norm(Gpp - np.swapaxes(Gpp, 1, 2))
    
                print("  symmetry error ||Gpp-Gpp.T_kl|| =", sym_error)
                print(
                    "  relative symmetry error         =",
                    sym_error / Gpp_norm if Gpp_norm != 0 else np.inf,
                )
                print(
                    "  estimated scale ||Gpp||*||p1||^2 =",
                    Gpp_norm * np.linalg.norm(p1) ** 2,
                )
    
                print("\nLargest qao3_term residual components")
                largest_m = np.argsort(np.abs(qao3_term))[-10:][::-1]
                for m in largest_m:
                    base_m = base_constant_terms[m]
                    ratio = qao3_term[m] / base_m if base_m != 0 else np.inf
                    print(
                        f"  m={m:6d}",
                        f"qao3={qao3_term[m]: .12e}",
                        f"base={base_m: .12e}",
                        f"qao3/base={ratio: .12e}",
                    )
    
                m_star = np.argmax(np.abs(qao3_term))
                pair_contrib = -Gpp[m_star] * np.outer(p1, p1)
    
                print("\nDominant parameter-pair contributions")
                print("  inspecting residual index m* =", m_star)
                print("  qao3_term[m*] =", qao3_term[m_star])
                print("  sum(pair_contrib) =", np.sum(pair_contrib))
    
                flat_idx = np.argsort(np.abs(pair_contrib.ravel()))[-20:][::-1]
                for idx in flat_idx:
                    k, l = np.unravel_index(idx, pair_contrib.shape)
                    print(
                        f"  k={k:6d}, l={l:6d}",
                        f"contrib={pair_contrib[k, l]: .12e}",
                        f"p1[k]={p1[k]: .12e}",
                        f"p1[l]={p1[l]: .12e}",
                        f"Gpp[m*,k,l]={Gpp[m_star, k, l]: .12e}",
                    )
    
                print_footer()
    
                constant_terms += qao3_term
    
            # ---------------------------------------------------------------------
            # Order 3 QAO=3 terms:
            #
            # -3 G_pp[p2,p1]
            # -3 E1 G_Epp[p1,p1]
            # -3 G_lpp[p1,p1]
            # ---------------------------------------------------------------------
            elif self.order == 3:
                p2 = self.previous_responses[1][:-1]
                E1 = e1
    
                Gpp = self.fanpt_container.d2_g_wfnparams2
                Gepp = self.fanpt_container.d3_g_e_wfnparams2
                Glpp = self.fanpt_container.d3_g_lambda_wfnparams2
    
                qao3_pp = -3.0 * np.einsum(
                    "mkl,k,l->m",
                    Gpp,
                    p2,
                    p1,
                )
    
                qao3_epp = -3.0 * E1 * np.einsum(
                    "mkl,k,l->m",
                    Gepp,
                    p1,
                    p1,
                )
    
                qao3_lpp = -3.0 * np.einsum(
                    "mkl,k,l->m",
                    Glpp,
                    p1,
                    p1,
                )
    
                base_norm = np.linalg.norm(base_constant_terms)
    
                print_header("ORDER 3 QAO3 DIAGNOSTICS")
    
                print("\nConstant-term norms")
                print("  ||base||     =", base_norm)
                print("  ||qao3_pp||  =", np.linalg.norm(qao3_pp))
                print("  ||qao3_epp|| =", np.linalg.norm(qao3_epp))
                print("  ||qao3_lpp|| =", np.linalg.norm(qao3_lpp))
    
                print(
                    "  pp/base      =",
                    np.linalg.norm(qao3_pp) / base_norm if base_norm != 0 else np.inf,
                )
                print(
                    "  epp/base     =",
                    np.linalg.norm(qao3_epp) / base_norm if base_norm != 0 else np.inf,
                )
                print(
                    "  lpp/base     =",
                    np.linalg.norm(qao3_lpp) / base_norm if base_norm != 0 else np.inf,
                )
    
                print_vector_diagnostics("p1", p1)
                print_vector_diagnostics("p2", p2)
    
                print("\nEnergy response diagnostics")
                print("  E1 =", E1)
    
                print_footer()
    
                constant_terms += qao3_pp + qao3_epp + qao3_lpp
    
        self.constant_terms = constant_terms
        return constant_terms
    















































    
