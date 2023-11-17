from numpy import abs
from numpy import arange
from numpy import concatenate
from numpy import float64
from numpy import insert
from numpy import ndarray
from numpy import zeros

from abc import ABC
from abc import abstractmethod

from .exceptions import LinearAlgebraError


class Solver(ABC):
    def __init__(self):
        self.coefficient_matrix = None
        self.scalars = None
        self.augmented_matrix = None
        self.solution_matrix = None

    @staticmethod
    def _apply_partial_pivoting(matrix: ndarray, iteration: int, rows: int):
        """
        Performs partial pivoting to avoid division by zero.

            :param matrix: Augmented matrix to which partial pivoting would be applied;
            :param iteration: current iteration over rows of the given matrix;
            :param rows: number of rows of the given matrix.
            :return: Given matrix with partial pivoting applied to it.
        """

        for row in arange(iteration + 1, rows):
            if abs(matrix[iteration, iteration]) < abs(matrix[row, iteration]):
                matrix[[row, iteration]] = matrix[[iteration, row]]

        if matrix[iteration, iteration] == 0.0:
            raise ZeroDivisionError(
                "Given system of linear equations is degenerate"
            )
        return matrix

    def fit(self, coefficient_matrix: ndarray, scalars: ndarray):
        """
        Checks if given coefficient_matrix is consistent and independent and
        scalars matrix has only one column.

            :param coefficient_matrix: Matrix of coefficients;
            :param scalars: column matrix of scalars.
        """

        if coefficient_matrix.shape[0] != coefficient_matrix.shape[1]:
            raise LinearAlgebraError(
                "Matrix A should be square"
            )

        if scalars.shape[1] > 1 or scalars.shape[0] != scalars.shape[0]:
            raise LinearAlgebraError(
                "Matrix B should be a column matrix with the same number of rows as matrix A"
            )

        self.coefficient_matrix, self.scalars = coefficient_matrix, scalars
        self.augmented_matrix = concatenate(
            (self.coefficient_matrix, self.scalars), axis=1, dtype=float64
        )

    @abstractmethod
    def solve(self, round_to: int):
        pass


class GaussianEliminator(Solver):
    """Solve systems of linear equations using Gaussian Elimination method with partial pivoting."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def _eliminate(matrix: ndarray, iteration: int, rows: int):
        """
        Performs Gaussian Elimination algorythm.

            :param matrix: Augmented matrix to which Gaussian Elimination would be applied;
            :param iteration: current iteration over rows of the given matrix;
            :param rows: number of rows of the given matrix.
            :return: Given matrix with Gaussian Elimination applied to it.
        """

        for row in arange(iteration + 1, rows):
            scaling_factor = matrix[row][iteration] / matrix[iteration][iteration]
            matrix[row] = matrix[row] - scaling_factor * matrix[iteration]
        return matrix

    @staticmethod
    def _substitute_backwards(solutions: ndarray, matrix: ndarray, rows: int):
        """
        Performs backward substitution to compute solution matrix.

            :param solutions: Input solution matrix with the last element being resolved;
            :param matrix: augmented matrix in the triangular form;
            :param rows: number of rows in the augmented matrix.
            :return: Solution matrix.
        """

        for row in arange(rows - 2, -1, -1):
            solutions[row] = matrix[row][rows]

            for column in arange(row + 1, rows):
                solutions[row] = solutions[row] - matrix[row][column] * solutions[column]
            solutions[row] = solutions[row] / matrix[row][row]

        return solutions

    def fit(self, coefficient_matrix: ndarray, scalars: ndarray):
        super().fit(coefficient_matrix, scalars)

    def solve(self, round_to: int):
        """
        Solves the system of linear equations passed to the fit() method.

            :param round_to: Number of decimals to which solution should be rounded.
            :return: Solution matrix for the given system of linear equations.
        """

        augmented_matrix = self.augmented_matrix
        rows = len(self.scalars)
        last = rows - 1
        iteration = 0
        solution_matrix = zeros(rows)

        while iteration < rows:
            augmented_matrix = self._apply_partial_pivoting(
                augmented_matrix, iteration=iteration, rows=rows
            )

            augmented_matrix = self._eliminate(
                augmented_matrix, iteration=iteration, rows=rows
            )
            iteration = iteration + 1

        solution_matrix[last] = augmented_matrix[last][rows] / augmented_matrix[last][last]

        solution_matrix = self._substitute_backwards(
            solution_matrix, augmented_matrix, rows=rows
        )
        solution_matrix = solution_matrix.round(round_to)
        self.solution_matrix = solution_matrix.reshape(rows, 1)

        return self.solution_matrix


class SquaresMinimizer(Solver):
    """Solve systems of linear equations using Least Squares method."""

    def __init__(self):
        super().__init__()
        self.gaussian_eliminator = GaussianEliminator()

    def fit(self, coefficient_matrix: ndarray, scalars: ndarray):
        """
        Performs some necessary computations to transform given matrices in appropriate for GaussianEliminator form.

            :param coefficient_matrix: Matrix of coefficients;
            :param scalars: column matrix of scalars.
        """
        transposed = coefficient_matrix.T
        self.coefficient_matrix = transposed @ coefficient_matrix
        self.scalars = transposed @ scalars
        self.gaussian_eliminator.fit(self.coefficient_matrix, self.scalars)

    def solve(self, round_to: int):
        """
        Solves the transformed system of linear equations using Gaussian Elimination method.

            :param round_to: Number of decimals to which solution should be rounded.
            :return: Solution matrix for the given system of linear equations.
        """
        return self.gaussian_eliminator.solve(round_to)


class TridiagonalSolver(Solver):
    """Attempt to implement Thomas algorythm for tridiagonal systems of linear equation."""

    def __init__(self):
        super().__init__()
        self.run_trough_matrix_p = None
        self.run_trough_matrix_q = None

    @staticmethod
    def _assign_diagonals(coefficient_matrix: ndarray, last: int):
        """
        Parses the given coefficient matrix to get diagonal vectors.

            :param coefficient_matrix: Matrix of coefficients;
            :param last: index of the last row of the coefficient matrix.
            :return: Tuple of diagonals parsed.
        """
        alpha, beta, gamma = (
            insert(coefficient_matrix.diagonal(-1), obj=0, values=.0),
            coefficient_matrix.diagonal(0),
            insert(coefficient_matrix.diagonal(1), obj=last, values=0.0)
        )
        return alpha, beta, gamma

    @staticmethod
    def _apply_recurrent_formula(
            alpha: ndarray, beta: ndarray, gamma: ndarray,
            coefficient_e: ndarray, coefficient_f: ndarray, last: int
    ):
        """
        Applies recurrent formula to get corresponding values of e and f for the computational part.

            :param alpha: Coefficient alpha;
            :param beta: coefficient beta;
            :param gamma: coefficient gamma;
            :param coefficient_e: solving coefficient e;
            :param coefficient_f: solving coefficient f;
            :param last: index of the last row of the coefficient matrix.
            :return: Solving coefficient matrices e and f.
        """
        for row in arange(1, last):
            denominator = beta[row] - alpha[row] * coefficient_e[row - 1]
            coefficient_e[row] = gamma[row] / denominator
            coefficient_f[row] = (
                (beta[row] - alpha[row] * coefficient_f[row - 1]) /
                denominator
            )
        return coefficient_e, coefficient_f

    @staticmethod
    def _compute_backwards(
            solution_matrix: ndarray, rows: int, alpha: ndarray, beta: ndarray,
            coefficient_e: ndarray, coefficient_f: ndarray
    ):
        """
        Computes the solution matrix moving upwards.

            :param solution_matrix: Solution matrix in which the first row is computed;
            :param rows: number of rows;
            :param alpha: coefficient alpha;
            :param beta: coefficient beta;
            :param coefficient_e: solving coefficient e;
            :param coefficient_f: solving coefficient f;
            :return: The solution matrix.
        """
        solution_matrix[rows - 1] = (
                (alpha[rows - 1] * coefficient_f[rows - 2] - beta[rows - 1]) /
                (alpha[rows - 1] * coefficient_e[rows - 2] - beta[rows - 1])
        )
        for row in arange(rows - 2, -1, -1):
            solution_matrix[row] = (
                    coefficient_f[row] - coefficient_e[row] * solution_matrix[row + 1]
            )
        return solution_matrix

    def fit(self, coefficient_matrix: ndarray, scalars: ndarray):
        super().fit(coefficient_matrix, scalars)

    def solve(self, round_to: int):
        scalars = self.scalars
        rows = len(self.scalars)
        last = rows - 1
        coefficient_matrix = self.coefficient_matrix
        coefficient_alpha, coefficient_beta, coefficient_gamma = (
            self._assign_diagonals(coefficient_matrix, last=last)
        )
        coefficient_e, coefficient_f = zeros(rows), zeros(rows)
        solution_matrix = zeros(rows)

        coefficient_e[0] = coefficient_gamma[0] / coefficient_beta[0]
        coefficient_f[0] = scalars[0] / coefficient_beta[0]

        coefficient_e, coefficient_f = (
            self._apply_recurrent_formula(
                alpha=coefficient_alpha, beta=coefficient_beta, gamma=coefficient_gamma,
                coefficient_e=coefficient_e, coefficient_f=coefficient_f, last=last
            )
        )

        solution_matrix = (
            self._compute_backwards(
                solution_matrix, rows=rows, alpha=coefficient_alpha, beta=coefficient_beta,
                coefficient_e=coefficient_e, coefficient_f=coefficient_f
            )
        )
        solution_matrix = solution_matrix.round(round_to).reshape(rows, 1)
        self.solution_matrix = solution_matrix

        return self.solution_matrix
