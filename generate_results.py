"""Generate reproducible figures without opening graphical windows."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import simulate_p2_3_4A as scenarios
import simulate_p4B_pso as trim
from rcam_model import xdot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'results')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 150})
    plt.show = lambda: None

    def save(name):
        fig = plt.gcf()
        fig.savefig(args.output / name, bbox_inches='tight', metadata={'Software': 'RCAM-Simulation'})
        plt.close(fig)

    manifest = {'seed': args.seed, 'duration_seconds': 180, 'integration_step_seconds': 0.05,
                'output_step_seconds': 1, 'scenario_failed_engine': 2, 'pso_failed_engine': 1,
                'versions': {name: importlib.metadata.version(name) for name in ['numpy', 'matplotlib', 'Pillow', 'pyswarms']},
                'source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                  for name in ['rcam_model.py', 'simulate_p2_3_4A.py', 'simulate_p4B_pso.py', 'generate_results.py']},
                'scenarios': {}}
    scenarios.FAILED_ENGINE = 2
    for case in [2, 3, 4]:
        print(f'Generating scenario {case}', flush=True)
        t, x, position, controls = scenarios.simulate_rcam(case, scenarios.X0, tf=180, dt_out=1, dt_int=0.05)
        if not all(np.isfinite(a).all() for a in [t, x, position, controls]):
            raise ValueError(f'Nonfinite output in scenario {case}')
        scenarios.plot_states(t, x, case)
        save(f'simulation_{case}_states.png')
        scenarios.plot_controls(t, controls, case)
        save(f'simulation_{case}_controls.png')
        scenarios.plot_trajectory_2d(position, case)
        save(f'simulation_{case}_trajectory_2d.png')
        scenarios.plot_trajectory_3d(position, case)
        save(f'simulation_{case}_trajectory_3d.png')
        np.savetxt(args.output / f'simulation_{case}.csv', np.column_stack([t, x, position, controls]), delimiter=',',
                   header='t,u,v,w,p,q,r,phi,theta,psi,north,east,down,aileron,elevator,rudder,throttle1,throttle2', comments='')
        manifest['scenarios'][str(case)] = {'samples': len(t), 'final_state': x[-1].tolist(), 'final_position_ned': position[-1].tolist()}

    print('Running PSO: 45 particles, 120 iterations', flush=True)
    np.random.seed(args.seed)
    trim.FAILED_ENGINE = 1
    best_cost, best, optimizer = trim.run_pso()
    x0, u0 = trim.build_state_control(best)
    residual = xdot(x0, u0)
    t, x = trim.simulate_trim(x0, u0)
    if not all(np.isfinite(a).all() for a in [x, residual, optimizer.cost_history]) or not np.isfinite(best_cost):
        raise ValueError('Nonfinite PSO result')
    trim.plot_cost_history(optimizer)
    save('pso_convergence.png')
    trim.plot_trim_states(t, x)
    save('pso_states.png')
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, np.linalg.norm(x[:, :3], axis=1), label='Simulated airspeed')
    ax.axhline(trim.TARGET_SPEED, color='tab:red', linestyle='--', label='Target: 78 m/s')
    ax.set(xlabel='Time [s]', ylabel='Airspeed [m/s]', title=f'Open-loop response after PSO search — seed {args.seed}')
    ax.grid(True, alpha=0.3)
    ax.legend()
    save('pso_airspeed.png')
    np.savetxt(args.output / 'pso_response.csv', np.column_stack([t, x]), delimiter=',', header='t,u,v,w,p,q,r,phi,theta,psi', comments='')
    np.savetxt(args.output / 'pso_cost.csv', np.array(optimizer.cost_history), delimiter=',', header='best_cost', comments='')
    manifest['pso'] = {'particles': 45, 'iterations': 120, 'target_speed': trim.TARGET_SPEED,
                       'best_cost': float(best_cost), 'initial_state': x0.tolist(), 'controls': u0.tolist(),
                       'initial_derivatives': residual.tolist(), 'final_airspeed': float(np.linalg.norm(x[-1, :3])),
                       'note': 'One stochastic search; not proof of global optimum or stable trim.'}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('Saved 15 figures, five CSV files and the run manifest.', flush=True)


if __name__ == '__main__':
    main()
