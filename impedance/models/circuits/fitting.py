import re  # <<< MODIFIED: needed by compileCircuit
import warnings

import numpy as np
from scipy.linalg import inv
from scipy.optimize import curve_fit, basinhopping

from .elements import circuit_elements, get_element_from_name

ints = '0123456789'


def rmse(a, b):
    """
    A function which calculates the root mean squared error
    between two vectors.

    Notes
    ---------
    .. math::

        RMSE = \\sqrt{\\frac{1}{n}(a-b)^2}
    """

    n = len(a)
    return np.linalg.norm(a - b) / np.sqrt(n)


def set_default_bounds(circuit, constants={}):
    """ This function sets default bounds for optimization.

    set_default_bounds sets bounds of 0 and np.inf for all parameters,
    except the CPE and La alphas which have an upper bound of 1.

    Parameters
    -----------------
    circuit : string
        String defining the equivalent circuit to be fit

    constants : dictionary, optional
        Parameters and their values to hold constant during fitting
        (e.g. {"RO": 0.1}). Defaults to {}

    Returns
    ------------
    bounds : 2-tuple of array_like
        Lower and upper bounds on parameters.
    """

    # extract the elements from the circuit
    extracted_elements = extract_circuit_elements(circuit)

    # loop through bounds
    lower_bounds, upper_bounds = [], []
    for elem in extracted_elements:
        raw_element = get_element_from_name(elem)
        for i in range(check_and_eval(raw_element).num_params):
            if elem in constants or elem + f'_{i}' in constants:
                continue
            if raw_element in ['CPE', 'La'] and i == 1:
                upper_bounds.append(1)
            else:
                upper_bounds.append(np.inf)
            lower_bounds.append(0)

    bounds = ((lower_bounds), (upper_bounds))
    return bounds


def circuit_fit(frequencies, impedances, circuit, initial_guess, constants={},
                bounds=None, weight_by_modulus=False, global_opt=False, pymoo_params=None,
                **kwargs):

    """ Main function for fitting an equivalent circuit to data.

    By default, this function uses `scipy.optimize.curve_fit
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html>`_
    to fit the equivalent circuit. This function generally works well for
    simple circuits. However, the final results may be sensitive to
    the initial conditions for more complex circuits. In these cases,
    the `scipy.optimize.basinhopping
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html>`_
    global optimization algorithm can be used to attempt a better fit.

    Parameters
    -----------------
    frequencies : numpy array
        Frequencies

    impedances : numpy array of dtype 'complex128'
        Impedances

    circuit : string
        String defining the equivalent circuit to be fit

    initial_guess : list of floats
        Initial guesses for the fit parameters

    constants : dictionary, optional
        Parameters and their values to hold constant during fitting
        (e.g. {"RO": 0.1}). Defaults to {}

    bounds : 2-tuple of array_like, optional
        Lower and upper bounds on parameters. Defaults to bounds on all
        parameters of 0 and np.inf, except the CPE alpha
        which has an upper bound of 1

    weight_by_modulus : bool, optional
        Uses the modulus of each data (|Z|) as the weighting factor.
        Standard weighting scheme when experimental variances are unavailable.
        Only applicable when global_opt = False

    global_opt : bool, optional
        If global optimization should be used (uses the basinhopping
        algorithm). Defaults to False

    kwargs :
        Keyword arguments passed to scipy.optimize.curve_fit or
        scipy.optimize.basinhopping

    Returns
    ------------
    p_values : list of floats
        best fit parameters for specified equivalent circuit

    p_errors : list of floats
        one standard deviation error estimates for fit parameters

    Notes
    ---------
    Need to do a better job of handling errors in fitting.
    Currently, an error of -1 is returned.

    """
    f = np.array(frequencies, dtype=float)
    Z = np.array(impedances, dtype=complex)

    # set upper and lower bounds on a per-element basis
    if bounds is None:
        bounds = set_default_bounds(circuit, constants=constants)

    if not global_opt and (pymoo_params is None):
        if 'maxfev' not in kwargs:
            kwargs['maxfev'] = 1e5
        if 'ftol' not in kwargs:
            kwargs['ftol'] = 1e-13

        # weighting scheme for fitting
        if weight_by_modulus:
            abs_Z = np.abs(Z)
            kwargs['sigma'] = np.hstack([abs_Z, abs_Z])

        popt, pcov = curve_fit(wrapCircuit(circuit, constants), f,
                               np.hstack([Z.real, Z.imag]),
                               p0=initial_guess, bounds=bounds, **kwargs)
    
    elif pymoo_params is not None:
        from pymoo.operators.sampling.lhs import LHS
        from pymoo.optimize import minimize
        from pymoo.core.problem import Problem
        

        # Define the optimization problem for pymoo
        # >>> MODIFIED (vectorized pymoo support) - START
        # Circuit is compiled once and the whole population is evaluated
        # in a single batched call instead of one wrapCircuit call per
        # individual.
        class CircuitProblem(Problem):
            def __init__(self, circuit, f, Z, constants):
                super().__init__(n_var=pymoo_params["n_par"],
                                 n_obj=1,
                                 n_constr=0,
                                 xl=pymoo_params["LB"],
                                 xu=pymoo_params["UB"])
                self.f = np.asarray(f)
                self.y = np.hstack([Z.real, Z.imag])
                try:
                    self.Z_fn = compileCircuit(circuit, constants)
                except KeyError:  # element without a batched implementation
                    self.Z_fn = None
                    self.model = wrapCircuit(circuit, constants)

            def _evaluate(self, x, out, *args, **kwargs):
                # scale x as an exponent of 10 
                x_corr = 10 ** np.array(x)
                if self.Z_fn is None:
                    out["F"] = np.array([rmse(self.model(self.f, *xi_corr), self.y)
                                         for xi_corr in x_corr])
                    return
                Zm = self.Z_fn(x_corr, self.f)
                err = np.hstack([Zm.real, Zm.imag]) - self.y
                out["F"] = np.sqrt(np.mean(err ** 2, axis=1))

        # >>> MODIFIED (vectorized pymoo support) - END
        problem = CircuitProblem(circuit, f, Z, constants)  # >>> MODIFIED: matches the new __init__ signature
        
        if pymoo_params["algorithm"] == "DE":
            from pymoo.algorithms.soo.nonconvex.de import DE
            algorithm = DE(
                            pop_size=pymoo_params['pop_size'],
                            sampling=LHS(),
                            variant="DE/rand/1/bin",
                            CR=pymoo_params['CR'],
                            dither="vector",
                            jitter=False,
                            n_gen=pymoo_params['n_gen']
                        )
        
        elif pymoo_params["algorithm"] == "GA":
            from pymoo.algorithms.soo.nonconvex.ga import GA
            algorithm = GA(
                            pop_size=pymoo_params['pop_size'],
                            sampling=LHS(),
                            n_gen=pymoo_params['n_gen'],
                        )
        elif pymoo_params["algorithm"] == "NRBO":
            from pymoo.algorithms.soo.nonconvex.nrbo import NRBO
            if 'deciding_factor' not in pymoo_params:
                raise ValueError("deciding_factor must be specified for NRBO algorithm")
            if 'max_iteration' not in pymoo_params:
                raise ValueError("max_iteration must be specified for NRBO algorithm")
            algorithm = NRBO(
                            pop_size=pymoo_params['pop_size'],
                            deciding_factor=pymoo_params['deciding_factor'],
                            max_iteration=pymoo_params['max_iteration'],
                            n_gen=pymoo_params['n_gen']
                        )
        
        elif pymoo_params["algorithm"] == "PSO":
            from pymoo.algorithms.soo.nonconvex.pso import PSO
            algorithm = PSO(
                            pop_size=pymoo_params['pop_size'],
                            adaptive=True,
                            sampling=LHS(),
                            n_gen=pymoo_params['n_gen']
                        )
        
        # >>> MODIFIED: algorithm passed positionally (method= is not accepted
        # by pymoo 0.6.2) and popt = res.X (res.X[0] was only the 1st parameter)
        # >>> MODIFIED: n_gen is not a DE argument (it was silently ignored),
        # so it is passed as the termination criterion instead
        res = minimize(problem, algorithm, ("n_gen", pymoo_params['n_gen']),
                       seed=0, verbose=True)
        popt_G = 10 ** np.array(res.X)
        # Apply local search 
        if 'maxfev' not in kwargs:
            kwargs['maxfev'] = 1e4
        if 'ftol' not in kwargs:
            kwargs['ftol'] = 1e-13

        # weighting scheme for fitting
        if weight_by_modulus:
            abs_Z = np.abs(Z)
            kwargs['sigma'] = np.hstack([abs_Z, abs_Z])

        popt, pcov = curve_fit(wrapCircuit(circuit, constants), f,
                                np.hstack([Z.real, Z.imag]),
                                p0=popt_G, bounds=bounds, **kwargs)
        
        
        # >>> MODIFIED: pymoo gives no covariance matrix, so perror is None
        # (np.sqrt(np.diag(None)) would crash)
        perror = None

    else:
        if 'seed' not in kwargs:
            kwargs['seed'] = 0

        def opt_function(x):
            """ Short function for basinhopping to optimize over.
            We want to minimize the RMSE between the fit and the data.

            Parameters
            ----------
            x : args
                Parameters for optimization.

            Returns
            -------
            function
                Returns a function (RMSE as a function of parameters).
            """
            return rmse(wrapCircuit(circuit, constants)(f, *x),
                        np.hstack([Z.real, Z.imag]))

        class BasinhoppingBounds(object):
            """ Adapted from the basinhopping documetation
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html
            """

            def __init__(self, xmin, xmax):
                self.xmin = np.array(xmin)
                self.xmax = np.array(xmax)

            def __call__(self, **kwargs):
                x = kwargs['x_new']
                tmax = bool(np.all(x <= self.xmax))
                tmin = bool(np.all(x >= self.xmin))
                return tmax and tmin

        basinhopping_bounds = BasinhoppingBounds(xmin=bounds[0],
                                                 xmax=bounds[1])
        results = basinhopping(opt_function, x0=initial_guess,
                               accept_test=basinhopping_bounds, **kwargs)
        popt = results.x

        # Calculate perror
        jac = results.lowest_optimization_result['jac'][np.newaxis]
        try:
            # jacobian -> covariance
            # https://stats.stackexchange.com/q/231868
            pcov = inv(np.dot(jac.T, jac)) * opt_function(popt) ** 2
            # covariance -> perror (one standard deviation
            # error estimates for fit parameters)
            perror = np.sqrt(np.diag(pcov))
        except (ValueError, np.linalg.LinAlgError):
            warnings.warn('Failed to compute perror')
            perror = None

    return popt, perror


# ##########################################################################
# >>> MODIFIED (vectorized pymoo support) - START
# Batched circuit evaluation used by the pymoo path of circuit_fit.
# Each p[k] is an (N, 1) array or a scalar constant and w has shape (M,),
# so every element returns an (N, M) array.
# ##########################################################################
VEC_ELEMENTS = {
    "R": lambda p, w: p[0] + 0j * w,
    "C": lambda p, w: 1.0 / (p[0] * 1j * w),
    "L": lambda p, w: p[0] * 1j * w,
    "W": lambda p, w: p[0] * (1 - 1j) / np.sqrt(w),
    "Wo": lambda p, w: p[0] / (np.sqrt(1j * w * p[1])
                               * np.tanh(np.sqrt(1j * w * p[1]))),
    "Ws": lambda p, w: p[0] * np.tanh(np.sqrt(1j * w * p[1]))
    / np.sqrt(1j * w * p[1]),
    "CPE": lambda p, w: 1.0 / (p[0] * (1j * w) ** p[1]),
    "La": lambda p, w: (p[0] * 1j * w) ** p[1],
}


def _split_top(s, sep):
    """ splits s by sep outside of any parentheses """
    parts, depth, cur = [], 0, ""
    for ch in s:
        depth += (ch == "(") - (ch == ")")
        if ch == sep and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    return parts + [cur]


def compileCircuit(circuit, constants=None):
    """ Parses a circuit string once and returns func(X, f) that evaluates
    the impedance for a whole population X of shape (N, n_free_params) at
    frequencies f, returning a complex array of shape (N, len(f)).

    Raises KeyError if the circuit contains an element without a batched
    implementation in VEC_ELEMENTS.
    """
    constants = constants or {}
    counter = [0]

    def build(expr):
        parts = _split_top(expr, "-")
        if len(parts) > 1:
            subs = [build(x) for x in parts]
            return lambda P, w: sum(s(P, w) for s in subs)
        if expr.startswith("p(") and expr.endswith(")"):
            subs = [build(x) for x in _split_top(expr[2:-1], ",")]
            return lambda P, w: 1.0 / sum(1.0 / s(P, w) for s in subs)
        typ = re.match(r"[A-Za-z]+", expr).group()
        fn = VEC_ELEMENTS[typ]
        n = circuit_elements[typ].num_params
        slots = []
        for j in range(n):
            key = f"{expr}_{j}" if n > 1 else expr
            if key in constants:
                slots.append(("c", constants[key]))
            else:
                slots.append(("x", counter[0]))
                counter[0] += 1
        return lambda P, w: fn(
            [v if k == "c" else P[:, [v]] for k, v in slots], w)

    func = build(circuit.replace(" ", ""))
    return lambda X, f: func(np.atleast_2d(X), 2 * np.pi * np.asarray(f))
# ##########################################################################
# >>> MODIFIED (vectorized pymoo support) - END
# ##########################################################################


def wrapCircuit(circuit, constants):
    """ wraps function so we can pass the circuit string """
    def wrappedCircuit(frequencies, *parameters):
        """ returns a stacked array of real and imaginary impedance
        components

        Parameters
        ----------
        circuit : string
        constants : dict
        parameters : list of floats
        frequencies : list of floats

        Returns
        -------
        array of floats

        """

        x = eval(buildCircuit(circuit, frequencies, *parameters,
                              constants=constants, eval_string='',
                              index=0)[0],
                 circuit_elements)
        y_real = np.real(x)
        y_imag = np.imag(x)

        return np.hstack([y_real, y_imag])
    return wrappedCircuit


def buildCircuit(circuit, frequencies, *parameters,
                 constants=None, eval_string='', index=0):
    """ recursive function that transforms a circuit, parameters, and
    frequencies into a string that can be evaluated

    Parameters
    ----------
    circuit: str
    frequencies: list/tuple/array of floats
    parameters: list/tuple/array of floats
    constants: dict

    Returns
    -------
    eval_string: str
        Python expression for calculating the resulting fit
    index: int
        Tracks parameter index through recursive calling of the function
    """

    parameters = np.array(parameters).tolist()
    frequencies = np.array(frequencies).tolist()
    circuit = circuit.replace(' ', '')

    def parse_circuit(circuit, parallel=False, series=False):
        """ Splits a circuit string by either dashes (series) or commas
            (parallel) outside of any paranthesis. Removes any leading 'p('
            or trailing ')' when in parallel mode """

        assert parallel != series, \
            'Exactly one of parallel or series must be True'

        def count_parens(string):
            return string.count('('), string.count(')')

        if parallel:
            special = ','
            if circuit.endswith(')') and circuit.startswith('p('):
                circuit = circuit[2:-1]
        if series:
            special = '-'

        split = circuit.split(special)

        result = []
        skipped = []
        for i, sub_str in enumerate(split):
            if i not in skipped:
                if '(' not in sub_str and ')' not in sub_str:
                    result.append(sub_str)
                else:
                    open_parens, closed_parens = count_parens(sub_str)
                    if open_parens == closed_parens:
                        result.append(sub_str)
                    else:
                        uneven = True
                        while i < len(split) - 1 and uneven:
                            sub_str += special + split[i+1]

                            open_parens, closed_parens = count_parens(sub_str)
                            uneven = open_parens != closed_parens

                            i += 1
                            skipped.append(i)
                        result.append(sub_str)
        return result

    parallel = parse_circuit(circuit, parallel=True)
    series = parse_circuit(circuit, series=True)

    if series is not None and len(series) > 1:
        eval_string += "s(["
        split = series
    elif parallel is not None and len(parallel) > 1:
        eval_string += "p(["
        split = parallel
    elif series == parallel:  # only single element
        split = series

    for i, elem in enumerate(split):
        if ',' in elem or '-' in elem:
            eval_string, index = buildCircuit(elem, frequencies,
                                              *parameters,
                                              constants=constants,
                                              eval_string=eval_string,
                                              index=index)
        else:
            param_string = ""
            raw_elem = get_element_from_name(elem)
            elem_number = check_and_eval(raw_elem).num_params
            param_list = []
            for j in range(elem_number):
                if elem_number > 1:
                    current_elem = elem + '_{}'.format(j)
                else:
                    current_elem = elem

                if current_elem in constants.keys():
                    param_list.append(constants[current_elem])
                else:
                    param_list.append(parameters[index])
                    index += 1

            param_string += str(param_list)
            new = raw_elem + '(' + param_string + ',' + str(frequencies) + ')'
            eval_string += new

        if i == len(split) - 1:
            if len(split) > 1:  # do not add closing brackets if single element
                eval_string += '])'
        else:
            eval_string += ','

    return eval_string, index


def extract_circuit_elements(circuit):
    """ Extracts circuit elements from a circuit string.

    Parameters
    ----------
    circuit : str
        Circuit string.

    Returns
    -------
    extracted_elements : list
        list of extracted elements.

    """
    p_string = [x for x in circuit if x not in 'p(),-']
    extracted_elements = []
    current_element = []
    length = len(p_string)
    for i, char in enumerate(p_string):
        if char not in ints:
            current_element.append(char)
        else:
            # min to prevent looking ahead past end of list
            if p_string[min(i+1, length-1)] not in ints:
                current_element.append(char)
                extracted_elements.append(''.join(current_element))
                current_element = []
            else:
                current_element.append(char)
    extracted_elements.append(''.join(current_element))
    return extracted_elements


def calculateCircuitLength(circuit):
    """ Calculates the number of elements in the circuit.

    Parameters
    ----------
    circuit : str
        Circuit string.

    Returns
    -------
    length : int
        Length of circuit.

    """
    length = 0
    if circuit:
        extracted_elements = extract_circuit_elements(circuit)
        for elem in extracted_elements:
            raw_element = get_element_from_name(elem)
            num_params = check_and_eval(raw_element).num_params
            length += num_params
    return length


def check_and_eval(element):
    """ Checks if an element is valid, then evaluates it.

    Parameters
    ----------
    element : str
        Circuit element.

    Raises
    ------
    ValueError
        Raised if an element is not in the list of allowed elements.

    Returns
    -------
    Evaluated element.

    """
    allowed_elements = circuit_elements.keys()
    if element not in allowed_elements:
        raise ValueError(f'{element} not in ' +
                         f'allowed elements ({allowed_elements})')
    else:
        return eval(element, circuit_elements)
