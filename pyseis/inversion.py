class Fwi():

  def __init__(wave_eq_solver, data, num_iter, model_bounds=None):
    # create the fwi operator which includes the nonlinear and linearized wave eq
    self.fwi_op = self._make_fwi_op(wave_eq_solver)

    # create inverse problem
    self.wave_eq_solver.sep_data.getNdArray()[:] = data
    self.problem = self._make_problem(self.survey.wave_equation.sep_model,
                                      self.survey.wave_equation.sep_data,
                                      self.fwi_op,
                                      model_bounds=model_bounds)

    # create solver
    self.solver = self._make_solver(config['inversion'])

  def _make_fwi_op(self, wave_eq_solver):
    nonlinear_op = wave_eq_solver.wave_prop_cpp_op
    linearized_op = wave_eq_solver.linearized_wave_prop_cpp_op

    return Operator.NonLinearOperator(nonlinear_op, linearized_op,
                                      linearized_op.set_background)

  def _make_problem(self, sep_model, sep_data, fwi_op, model_bounds=None):
    if model_bounds is not None:
      min_bound = model_bounds[0]
      max_bound = model_bounds[1]

    return Prblm.ProblemL2NonLinear(sep_model,
                                    sep_data,
                                    fwi_op,
                                    minBound=min_bound,
                                    maxBound=max_bound)
