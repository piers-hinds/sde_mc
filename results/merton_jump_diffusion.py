from sde_mc import *
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # Model params
    parser.add_argument('--r', default=0.05, type=float)
    parser.add_argument('--sigma', default=0.3, type=float)
    parser.add_argument('--rate', default=2, type=float)
    parser.add_argument('--alpha', default=-0.05, type=float)
    parser.add_argument('--gamma', default=0.3, type=float)
    parser.add_argument('--spot', default=1, type=float)
    
    # Scheme params
    parser.add_argument('--expiry', default=3, type=float)
    parser.add_argument('--steps', default=3000, type=int)
    parser.add_argument('--trials', default=10000, type=int)
    
    # Config
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=1, type=int)
    
    arg = parser.parse_args()

    # Model params:
    r = arg.r
    vol = arg.sigma
    alpha = arg.alpha
    gamma = arg.gamma
    rate = arg.rate
    spot = arg.spot

    x0 = torch.tensor([spot], device=device)
    jump_mean = (np.exp(alpha + 0.5*gamma**2)) - 1

    # Scheme params
    expiry = arg.expiry
    steps = arg.steps
    trials = arg.trials
    time_points = partition(expiry, steps, device=device)

    # Option and discounter
    euro_call = EuroCall(strike=1.)
    discounter = ConstantShortRate(r=r)
    Ys = discounter(time_points)

    # Merton SDE and SdeSolver
    pure_jump = Merton(r - rate*jump_mean, vol, x0, rate, alpha, gamma)
    dim = pure_jump.dim
    solver = JumpSolver(pure_jump, expiry, steps, device=device)

    # Analytical price
    true_price = merton_jump_call(spot, 1, expiry, r, vol, alpha, gamma, rate)
    print(round(true_price, 6))
    
    # Define modules for each control variate
    brownian_control_variate = Mlp(dim + 1, [50, 50, 50], 1, activation=nn.ReLU).to(device)
    jumps_control_variate = Mlp(dim + 1, [50, 50, 50], 1, activation=nn.ReLU).to(device)

    # Joint optimizer
    adam = optim.Adam(list(brownian_control_variate.parameters()) + list(jumps_control_variate.parameters()))
    mc_stats = mc_control_variates([brownian_control_variate, jumps_control_variate], adam, solver, trials=(10000, 40000), steps=(300, 3000), 
                                 payoff=euro_call, discounter=discounter, jump_mean=jump_mean, bs=(1000, 1000), epochs=10)
    mc_stats.print()
