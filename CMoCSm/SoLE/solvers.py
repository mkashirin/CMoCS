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
        self.scalars_vector = None
        self.augmented_matrix = None
        self.solution_vector = None

    @staticmethod
    def _apply_partial_pivoting(matrix: ndarray, iteration: int, rows: int):
        """
        Perform partial pivoting to avoid division by zero.

            :param matrix: Augmented matrix to which partial pivoting would be applied;
            :param iteration: current iteration over rows of the given matrix;
            :param rows: number of rows in the given matrix.
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

    def fit(self, coefficient_matrix: ndarray, scalars_vector: ndarray):
        """
        Check if given coefficient_matrix is consistent, independent and scalars vector has only one column.

            :param coefficient_matrix: Matrix of coefficients;
            :param scalars_vector: column vector of scalars vector.
        """

        if coefficient_matrix.shape[0] != coefficient_matrix.shape[1]:
            raise LinearAlgebraError(
                "Matrix A should be square"
            )

        if scalars_vector.shape[1] > 1 or scalars_vector.shape[0] != scalars_vector.shape[0]:
            raise LinearAlgebraError(
                "Matrix B should be a column matrix with the same number of rows as matrix A"
            )

        self.coefficient_matrix, self.scalars_vector = coefficient_matrix, scalars_vector
        self.augmented_matrix = concatenate(
            (self.coefficient_matrix, self.scalars_vector), axis=1, dtype=float64
        )

    @abstractmethod
    def solve(self, round_to: int):
        pass


class GaussianEliminationSolver(Solver):
    """Solve independent systems of linear equations using Gaussian Elimination method with partial pivoting."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def _eliminate(matrix: ndarray, iteration: int, rows: int):
        """
        Perform Gaussian Elimination algorythm.

            :param matrix: Augmented matrix to which Gaussian Elimination would be applied;
            :param iteration: current iteration over rows of the given matrix;
            :param rows: number of rows in the given matrix.
            :return: Given matrix with Gaussian Elimination applied to it.
        """

        for row in arange(iteration + 1, rows):
            scaling_factor = matrix[row][iteration] / matrix[iteration][iteration]
            matrix[row] = matrix[row] - scaling_factor * matrix[iteration]
        return matrix

    @staticmethod
    def _substitute_backwards(scalars_vector: ndarray, matrix: ndarray, rows: int):
        """
        Substitute backwards to compute solution matrix.

            :param scalars_vector: Input solution matrix with the last element being resolved;
            :param matrix: augmented matrix in the triangular form;
            :param rows: number of rows in the augmented matrix.
            :return: Solution matrix.
        """

        for row in arange(rows - 2, -1, -1):
            scalars_vector[row] = matrix[row][rows]

            for column in arange(row + 1, rows):
                scalars_vector[row] = scalars_vector[row] - matrix[row][column] * scalars_vector[column]
            scalars_vector[row] = scalars_vector[row] / matrix[row][row]

        return scalars_vector

    def fit(self, coefficient_matrix: ndarray, scalars_vector: ndarray):
        super().fit(coefficient_matrix, scalars_vector)

    def solve(self, round_to: int):
        """
        Solve the independent system of linear equations.

            :param round_to: Number of decimals to which solution should be rounded.
            :return: Solution matrix for the given system of linear equations.
        """

        augmented_matrix = self.augmented_matrix
        rows = len(self.scalars_vector)
        index = rows - 1
        iteration = 0
        solution_vector = zeros(rows)

        while iteration < rows:
            augmented_matrix = self._apply_partial_pivoting(
                augmented_matrix, iteration=iteration, rows=rows
            )

            augmented_matrix = self._eliminate(
                augmented_matrix, iteration=iteration, rows=rows
            )
            iteration = iteration + 1

        solution_vector[index] = augmented_matrix[index][rows] / augmented_matrix[index][index]

        solution_vector = self._substitute_backwards(
            solution_vector, augmented_matrix, rows=rows
        )
        self.solution_vector = solution_vector.round(round_to).reshape(rows, 1)

        return self.solution_vector


class LeastSquaresSolver(Solver):
    """LeastSquaresSolver solves dependent systems of linear equations using Least Squares method."""

    def __init__(self):
        super().__init__()
        self.gaussian_eliminator = GaussianEliminationSolver()

    def fit(self, coefficient_matrix: ndarray, scalars_vector: ndarray):
        """
        Perform computations to transform given matrices in appropriate for GaussianEliminationSolver form.

            :param coefficient_matrix: Matrix of coefficients;
            :param scalars_vector: Column vector of scalars.
        """
        transposed = coefficient_matrix.T
        self.coefficient_matrix = transposed @ coefficient_matrix
        self.scalars_vector = transposed @ scalars_vector
        self.gaussian_eliminator.fit(self.coefficient_matrix, self.scalars_vector)

    def solve(self, round_to: int):
        """
        Solve the transformed system of linear equations using Gaussian Elimination method.

            :param round_to: Number of decimals to which solution should be rounded.
            :return: Solution vector for the given system of linear equations.
        """
        return self.gaussian_eliminator.solve(round_to)


class TridiagonalMatrixSolver(Solver):
    """TridiagonalSolver solves tridiagonal systems of linear equations using Tridiagonal Matrix Method."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def _get_diagonals(coefficient_matrix: ndarray, index: int):
        """
        Parse the given coefficient matrix to extract diagonals properly.

            :param coefficient_matrix: Matrix of coefficients;
            :param index: index of the last row in the coefficient matrix.
            :return: Tuple of diagonals parsed.
        """
        alpha, beta, gamma = (
                insert(coefficient_matrix.diagonal(-1), obj=0, values=0.0),
                coefficient_matrix.diagonal(0),
                insert(coefficient_matrix.diagonal(1), obj=index, values=0.0)
        )
        return alpha, beta, gamma

    @staticmethod
    def _run_forward(
            scalars_vector: ndarray, rows: int, alpha: ndarray, beta: ndarray, gamma: ndarray,
            coefficient_p: ndarray, coefficient_q: ndarray
    ):
        """
        Perform forward run-trough.

            :param scalars_vector: Column vector of scalars;
            :param rows: number of rows of the matrix;
            :param alpha: alpha;
            :param beta: beta;
            :param gamma: gamma;
            :param coefficient_p: P;
            :param coefficient_q: Q.
            :return: Run-trough coefficients matrices P and Q.
        """
        for row in arange(1, rows):
            denominator = beta[row] - alpha[row] * coefficient_p[row - 1]
            coefficient_p[row] = gamma[row] / denominator
            coefficient_q[row] = (
                    (scalars_vector[row] - alpha[row] * coefficient_q[row - 1]) /
                    denominator
            )
        return coefficient_p, coefficient_q

    @staticmethod
    def _run_backwards(
            solution_vector: ndarray, rows: int, coefficient_p: ndarray,
            coefficient_q: ndarray
    ):
        """
        Perform backward run-trough.

            :param solution_vector: Solution matrix in which the first row is already computed;
            :param coefficient_p: P;
            :param coefficient_q: Q.
            :return: The solution vector.
        """
        solution_vector[rows - 1] = coefficient_q[rows - 1]
        for row in arange(rows - 2, -1, -1):
            solution_vector[row] = (
                    coefficient_q[row] - coefficient_p[row] * solution_vector[row + 1]
            )
        return solution_vector

    def fit(self, coefficient_matrix: ndarray, scalars_vector: ndarray):
        super().fit(coefficient_matrix, scalars_vector)

    def solve(self, round_to: int):
        """
        Solve the independent system of linear equations.

            :param round_to: Number of decimals to which solution should be rounded.
            :return: Solution vector for the given system of linear equations.
        """
        scalars_vector = self.scalars_vector
        rows = len(self.scalars_vector)
        index = rows - 1
        coefficient_matrix = self.coefficient_matrix
        alpha, beta, gamma = (
            self._get_diagonals(coefficient_matrix, index=index)
        )
        coefficient_p, coefficient_q = zeros(rows), zeros(rows)
        solution_vector = zeros(rows)

        coefficient_p[0] = gamma[0] / beta[0]
        coefficient_q[0] = scalars_vector[0] / beta[0]

        coefficient_p, coefficient_q = (
            self._run_forward(
                scalars_vector,  rows=rows, alpha=alpha, beta=beta, gamma=gamma,
                coefficient_p=coefficient_p, coefficient_q=coefficient_q
            )
        )

        solution_vector = (
            self._run_backwards(
                solution_vector, rows=rows, coefficient_p=coefficient_p,
                coefficient_q=coefficient_q
            )
        )
        self.solution_vector = solution_vector.round(round_to).reshape(rows, 1)

        return self.solution_vector
