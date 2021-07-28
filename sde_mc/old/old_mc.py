def mc_control_variate(num_trials, simple_solver, approximator, payoff, discounter, step_factor=5, time_points=None,
                       bs=None):
    """Run Monte Carlo simulation of an SDE with a control variate

    :param num_trials: tuple (int, int)
        The number of trials for the approximation and the number of trials for the
        control variate monte carlo method

    :param simple_solver: SdeSolver
        The solver for the Sde with no control variate

    :param approximator: SdeApproximator
        The approximator for the solution of the Sde

    :param payoff: Option
        The payoff function applied to the terminal value - see the Option class

    :param discounter: function(float: time)
        The discount process to be applied to the payoff

    :param step_factor: int
        The factor to increase the number of steps used in the initial solver

    :param bs: int
        The batch size for the monte carlo method

    :return: MCStatistics
        The relevant statistics from the MC simulation - see the MCStatistics class
    """
    if time_points is None:
        time_points = approximator.time_points
    simple_trials, cv_trials = num_trials
    discount = discounter(torch.tensor(simple_solver.time))
    start = time.time()
    simple_stats = mc_simple(simple_trials, simple_solver, payoff, discount)
    approximator.fit(simple_stats.paths, simple_stats.payoffs)
    cv_sde = SdeControlVariate(base_sde=simple_solver.sde, control_variate=approximator,
                               time_points=time_points, discounter=discounter)
    cv_solver = SdeSolver(sde=cv_sde, time=3, num_steps=simple_solver.num_steps * step_factor,
                          device=simple_solver.device)

    def cv_payoff(spot):
        return discount * payoff(spot[:, :simple_solver.sde.dim]) + spot[:, simple_solver.sde.dim]

    cv_stats = mc_simple(cv_trials, cv_solver, cv_payoff, discount=1, bs=bs)
    print('Time for final MC:', cv_stats.time_elapsed)
    end = time.time()
    cv_stats.time_elapsed = end - start
    return cv_stats


def mc_min_variance(trials, solver, model, payoff, discounter, bs, step_factor=5, pre_trained=False, epochs=10):
    # Setup
    init_trials, final_trials = trials if not pre_trained else (0, trials)
    init_bs, final_bs = bs
    steps = solver.num_steps
    solver.num_steps = int(steps / step_factor)
    ts = torch.tensor([solver.time * i / solver.num_steps for i in range(solver.num_steps)], device=solver.device)
    discounts = discounter(ts)
    dim = solver.sde.dim
    end_time = torch.tensor(solver.time)

    start = time.time()
    # MC with no control variate on coarse grid
    mc_stats = mc_simple(num_trials=init_trials, sde_solver=solver, payoff=payoff,
                         discount=discounter(end_time), return_normals=True)
    mc_stats.print()
    paths, payoffs, normals = mc_stats.paths, mc_stats.payoffs, mc_stats.normals.squeeze(-1)

    # Set up data
    data = NormalPathData(paths, payoffs, normals)
    dl = DataLoader(data, batch_size=init_bs, shuffle=True, drop_last=True)
    adam = optim.Adam(model.parameters())

    # Train net
    train_control_variate(model, dl, adam, ts, dim, Ys=discounts, epochs=epochs)

    # New MC with finer time grid
    solver.num_steps = steps
    new_ts = torch.tensor([3 * i / solver.num_steps for i in range(solver.num_steps)], device=solver.device)
    new_discounts = discounter(new_ts)

    new_mc_stats = mc_simple(num_trials=final_trials, sde_solver=solver, payoff=payoff,
                             discount=discounter(end_time), return_normals=True)
    new_paths, new_payoffs, new_normals = new_mc_stats.paths, new_mc_stats.payoffs, new_mc_stats.normals.squeeze(-1)

    new_data = NormalPathData(new_paths, new_payoffs, new_normals)
    new_dl = DataLoader(new_data, batch_size=final_bs, shuffle=False)

    # Get predictions of control variate
    mn, sd = get_preds(model, new_dl, new_ts, dim, new_Ys=new_discounts)

    # Return mean, std
    end = time.time()
    tt = end - start
    return MCStatistics(mn, sd, tt)
